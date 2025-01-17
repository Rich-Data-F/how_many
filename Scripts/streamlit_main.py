import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
from prophet import Prophet
import io

# Set page title
st.title('Sales Prediction App')

# Function to check if model needs refresh
def check_model_freshness():
    model_path = '../Models/prophet_fit_models_units.pkl'
    if not os.path.exists(model_path):
        return True
    model_modified_time = datetime.fromtimestamp(os.path.getmtime(model_path))
    return model_modified_time.date() != datetime.now().date()

# Function to refresh model
def refresh_model():
    st.info("Refreshing model with latest data...")
    
    # Load and preprocess data
    df = pd.read_csv('../Data/transformed_data/historical_sales_data.csv')
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    
    # Load holidays data
    holidays_df = pd.read_csv('../Data/external_data/holidays.csv')
    holidays_df['Full_date'] = pd.to_datetime(holidays_df['Full_date'])
    holidays_df = holidays_df.rename(columns={
        'Full_date': 'ds',
        'National holiday': 'holiday'
    })
    
    # Train models for each article
    fit_models_units = {}
    unique_articles = df['Article_name'].unique()
    
    progress_bar = st.progress(0)
    for idx, name in enumerate(unique_articles):
        frame = df[df['Article_name']==name].copy()
        frame.drop(['Article_name', 'Price'], axis=1, inplace=True)
        frame.columns = ['ds', 'y']
        m = Prophet(interval_width=0.95, yearly_seasonality=True, holidays=holidays_df)
        model = m.fit(frame)
        fit_models_units[name] = m
        progress_bar.progress((idx + 1) / len(unique_articles))
    
    # Save updated models
    with open('../Models/prophet_fit_models_units.pkl', 'wb') as f:
        pickle.dump(fit_models_units, f)
    
    st.success("Model refresh complete!")
    return fit_models_units

# Check and refresh model if needed
if check_model_freshness():
    models = refresh_model()
else:
    with open('../Models/prophet_fit_models_units.pkl', 'rb') as f:
        models = pickle.load(f)

# Load categories from Excel file
categories_df = pd.read_excel('../Data/transformed_data/categorised_articles.xlsx')
categories = categories_df['Category'].unique()

# Create multiselect for categories
selected_categories = st.multiselect(
    'Select categories:',
    categories
)

# Get all products from selected categories
category_products = []
if selected_categories:
    category_products = categories_df[categories_df['Category'].isin(selected_categories)]['Article_name'].unique()

# Create multiselect for products within the selected categories
selected_products = st.multiselect(
    'Select/deselect products to predict:',
    category_products,
    default=category_products  # All products selected by default
)

# Create number input for days
num_days = st.number_input(
    'Number of days to predict ahead:',
    min_value=1,
    max_value=365,
    value=30
)

if st.button('Generate Predictions'):
    if selected_products:
        # Create a container for results
        results_container = st.container()
        
        # Initialize DataFrame for all predictions
        all_predictions = pd.DataFrame()
        
        with results_container:
            # Create a combined plot for all products
            combined_predictions = pd.DataFrame()
            
            for product in selected_products:
                # Get the model for this product
                model = models[product]
                
                # Create future dates dataframe
                future = model.make_future_dataframe(periods=num_days)
                
                # Make predictions
                forecast = model.predict(future)
                
                # Get the last num_days predictions
                results_df = forecast[['ds', 'yhat']].tail(num_days)
                results_df.columns = ['Date', 'Predicted_Sales']
                results_df['Product'] = product
                results_df['Predicted_Sales'] = results_df['Predicted_Sales'].round(2)
                
                # Add to all_predictions
                all_predictions = pd.concat([all_predictions, results_df])
                
                # Add to combined predictions for plotting
                combined_predictions = pd.concat([combined_predictions, results_df])
            
            # Plot all predictions on one graph
            st.subheader('Combined Predictions for All Selected Products')
            chart_data = combined_predictions.pivot(index='Date', columns='Product', values='Predicted_Sales')
            st.line_chart(chart_data)
            
            # Show individual product details
            for product in selected_products:
                st.subheader(f'Detailed Analysis for {product}')
                product_data = all_predictions[all_predictions['Product'] == product]
                st.dataframe(product_data)
                
                # Plot weekly and yearly trends
                model = models[product]
                future = model.make_future_dataframe(periods=num_days)
                forecast = model.predict(future)
                fig1 = model.plot_components(forecast)
                st.write(f"Weekly and Yearly Trends for {product}")
                st.pyplot(fig1)
        
        # Prepare Excel file for download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Pivot the data for better Excel format
            pivot_predictions = all_predictions.pivot(
                index='Date',
                columns='Product',
                values='Predicted_Sales'
            )
            pivot_predictions.to_excel(writer, sheet_name='Predictions')
        
        # Offer Excel file for download
        output.seek(0)
        current_date = datetime.now().strftime('%Y%m%d')
        st.download_button(
            label="Download Predictions as Excel",
            data=output,
            file_name=f'projections_{current_date}_decision_units.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.warning('Please select at least one product.')
