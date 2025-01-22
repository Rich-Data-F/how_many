import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
from io import BytesIO
import time
import logging

# Set page title and layout
st.set_page_config(page_title="Sales Analysis and Prediction App", layout="wide")

# Landing Page Header
st.title("Welcome to the Sales Analysis and Prediction App")
st.subheader("Analyze historical sales data and generate future predictions.")


# Load Data Paths
data_path = os.path.join(os.getcwd(), 'Data', 'transformed_data', 'historical_sales_data.csv')
categories_path = os.path.join(os.getcwd(), 'Data', 'transformed_data', 'categorised_articles.xlsx')
holidays_path = os.path.join(os.getcwd(), 'Data', 'external_data', 'holidays.csv')
# Define the path to the saved models
model_path = os.path.join(os.getcwd(), 'Models', 'fitted_models_results.pkl')
article_to_category_map = pd.read_excel(categories_path)
categories_df = pd.read_excel(categories_path)

# Initialize session state for historical data
if "historical_data" not in st.session_state:
    if os.path.exists(data_path):
        st.session_state.historical_data = pd.read_csv(data_path)
        st.session_state.historical_data['Date'] = pd.to_datetime(
            st.session_state.historical_data['Date'], format='mixed', dayfirst=True)
        st.session_state.latest_data_date = st.session_state.historical_data['Date'].max()
    else:
        st.error("Historical sales data not found. Please ensure the file exists.")
        st.stop()

# Load other necessary data files
if os.path.exists(categories_path):
    categories_df = pd.read_excel(categories_path)
else:
    st.error("Categorized articles file not found. Please ensure the file exists.")
    st.stop()

if os.path.exists(holidays_path):
    holidays_df = pd.read_csv(holidays_path)
    holidays_df['Full_date'] = pd.to_datetime(holidays_df['Full_date'], format='mixed', dayfirst=True)
    holidays_df.rename(columns={'Full_date': 'ds', 'National holiday': 'holiday'}, inplace=True)
else:
    st.error("Holidays data file not found. Please ensure the file exists.")
    st.stop()

### Set up logging ###
log_file_path = os.path.join(os.getcwd(), 'Logs', f'refresh_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

### Function Definitions ###

def load_historical_data():
# Load Historical Data
    if os.path.exists(data_path):
        st.session_state.historical_data['Date'] = pd.to_datetime(st.session_state.historical_data['Date'], format='mixed', dayfirst=True)
        st.session_state.latest_data_date = st.session_state.historical_data['Date'].max()
        st.write(f"Latest Historical Data Date: {st.session_state.latest_data_date.date()}")
    else:
        st.error("Historical sales data not found. Please ensure the file exists.")
        st.stop()
    return

def load_category_data():
# Load Categories Data
    if os.path.exists(categories_path):
        article_to_category_map = pd.read_excel(categories_path)
        categories_df = pd.read_excel(categories_path)
    else:
        st.error("Categorized articles file not found. Please ensure the file exists.")
        st.stop()

def map_categories_to_historical_data():
    """Map categories to historical sales data."""
    if "historical_data" not in st.session_state:
        st.error("Historical data is not loaded.")
        return
    
    # Copy historical data
    historical_data = st.session_state.historical_data.copy()

    # Merge with category data
    if 'Article_name' in st.session_state.historical_data.columns and 'Article_name' in article_to_category_map.columns:
        filtered_data = st.session_state.historical_data.merge(
            article_to_category_map[['Category', 'Article_name']], 
            on='Article_name', 
            how='left'
        )
        filtered_data['Date'] = pd.to_datetime(filtered_data['Date'], format='%d/%b/%Y', dayfirst=True, errors='coerce')

        # Store filtered data in session state
        st.session_state.filtered_data = filtered_data
    else:
        st.error("Required columns for merging are missing.")
    return filtered_data


def load_holidays_date():
    # Load Holidays Data
    if os.path.exists(holidays_path):
        holidays_df = pd.read_csv(holidays_path)
        holidays_df['Full_date'] = pd.to_datetime(holidays_df['Full_date'], format='mixed', dayfirst=True)
        holidays_df.rename(columns={'Full_date': 'ds', 'National holiday': 'holiday'}, inplace=True)
    else:
        st.error("Holidays data file not found. Please ensure the file exists.")
        st.stop()

def clean_article_names():
    # Merge the mapping with the all_pos_sales_units_prices dataframe
    mapping_path=os.path.join(os.getcwd(), 'Data', 'to_explore','unique_article_names_for_mapping.xlsx')
    mapping = pd.read_excel(mapping_path)
    historical_sales_units_prices = historical_sales_data.merge(mapping, left_on='Article_name',\
    right_on='Initial_article_name', how='left')
    # Replace the Article_name with the renamed_article
    historical_sales_units_prices['Article_name'] = historical_sales_units_prices['Renamed_article']
    # Drop the renamed_article column
    historical_sales_units_prices = historical_sales_units_prices.drop(columns=['Renamed_article', 'Initial_article_name'])
    historical_sales_units_prices = historical_sales_units_prices.groupby('Article_name','Date')[('Quantity','Price')].sum()
    print(historical_sales_units_prices.shape)

def random_sample_split(df, validation_fraction=0.2, seed=42):
    """
    Splits the dataset into training and validation sets using random sampling.
    """
    np.random.seed(seed)
    all_dates = df['ds'].unique()
    val_dates = np.random.choice(all_dates, size=int(len(all_dates) * validation_fraction), replace=False)
    val_df = df[df['ds'].isin(val_dates)]
    train_df = df[~df['ds'].isin(val_dates)]
    return train_df, val_df


# Utility function to save models
def save_models(models_dict, file_path):
    """
    Saves a dictionary of models to a specified file path.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(models_dict, f)
    st.success(f"Models saved successfully at {file_path}.")


# Utility function for progress bar updates

def autocalibrate_prophet(df, holidays_df, validation_fraction=0.2, seed=42):
    """Automatically calibrates a Prophet model
    This process includes a progress bar to visualize the calibration progress.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'ds' (date) and 'y' (target variable).
        holidays_df (pd.DataFrame): DataFrame containing holiday information.
        validation_fraction (float): Fraction of data to be used for validation.
        seed (int, optional): Random seed for reproducibility.
    Automatically calibrates a Prophet model by tuning hyperparameters using random sampling for validation.
    Returns:
        dict: A dictionary containing the best model, parameters, and MAE for each article.
    """
    if "filtered_data" not in st.session_state:
        st.error("Filtered data is not available. Please ensure categories are mapped.")
        return
    # Retrieve filtered data
    df = st.session_state.filtered_data.copy()    
    # Define hyperparameter grid
    param_grid = {
        'changepoint_prior_scale': [0.1, 1, 5],
        'seasonality_mode': ['additive', 'multiplicative'],
        'holidays_prior_scale': [0.1, 1, 5]
    }
    grid = ParameterGrid(param_grid)
    
    # Initialize results dictionary
    results = {}
    
    # Get unique articles and initialize progress bar
    unique_articles = df['Article_name'].unique()
    progress_bar = st.progress(0)

    for idx, article in enumerate(unique_articles):
        st.write(f"Processing article: {article} ({idx + 1}/{len(unique_articles)})")
        # Update progress bar
        progress_bar.progress((idx + 1) / len(unique_articles))
        
        # Filter data for the current article
        frame = df[df['Article_name'] == article].copy()
        frame.drop(['Article_name', 'Price', 'Category'], axis=1, inplace=True)
        frame.columns = ['ds', 'y']

        # Split data into training and validation sets
        train_data = frame.sample(frac=1 - validation_fraction, random_state=seed)
        val_data = frame.drop(train_data.index)

        best_mae = float('inf')
        best_params = None
        best_model = None

        # Iterate over all parameter combinations in the grid
        for params in grid:
            try:
                model = Prophet(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_mode=params['seasonality_mode'],
                    holidays=holidays_df,
                    holidays_prior_scale=params['holidays_prior_scale']
                )
                model.fit(train_data)

                # Make predictions on validation set
                future = val_data[['ds']]
                forecast = model.predict(future)

                y_true = val_data['y'].values
                y_pred = forecast['yhat'].values

                # Calculate MAE on validation set
                mae = mean_absolute_error(y_true, y_pred)
                mape = mean_absolute_percentage_error(y_true, y_pred)
                accuracy = (1 - mape) * 100  # Calculate accuracy as a percentage

                # Update best parameters if current MAE is lower
                if mae < best_mae:
                    best_mae = mae
                    best_params = params
                    best_model = model
                    best_accuracy = accuracy
                    best_mape = mape

            except Exception as e:
                logging.error(f"Error during model fitting: {e}")
                continue

        # Save the best model and parameters for the article with a timestamp
        results[article] = {
            'model': best_model,
            'params': best_params,
            'mae': best_mae,
            'accuracy': best_accuracy,
            'mape':best_mape,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save results to a file after processing each article
        results_file_path = os.path.join(os.getcwd(), 'Models', 'fitted_models_results.pkl')
        with open(results_file_path, 'wb') as f:
            pickle.dump(results, f)

        # Log the results of calibration for the current article
        logging.info(f"Processed article: {article}. Best MAE: {best_mae}, Best_model_MAPE: {best_mape}, Best model accuracy: {best_accuracy}, Best Params: {best_params}")

    # Final log after all articles are processed
    logging.info("Autocalibration complete for all articles.")

    st.success(f"Results saved successfully at {results_file_path}.")
    
    return results


def refresh_model(selected_categories=None, selected_articles=None):
    """
    Refreshes Prophet models with incremental updates for each article.
    
    Parameters:
        selected_categories (list): List of categories to filter data by.
        selected_articles (list): List of articles to filter data by.
    
    Returns:
        dict: A dictionary of trained Prophet models for each article.
    """
    st.info("Refreshing models with incremental updates...")

    # Load and preprocess data
    load_historical_data()
    load_category_data
    map_categories_to_historical_data()
    df=st.session_state.filtered_data
#    file_path = os.path.join(os.getcwd(), 'Data', 'transformed_data', 'historical_sales_data.csv')
#    df = pd.read_csv(file_path)
#    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    if selected_categories:
        df = df[df['Category'].isin(selected_categories)]
    
    if selected_articles:
        df = df[df['Article_name'].isin(selected_articles)]

    # Load holidays data
    holidays_path = os.path.join(os.getcwd(), 'Data', 'external_data', 'holidays.csv')
    holidays_df = pd.read_csv(holidays_path)
    holidays_df['Full_date'] = pd.to_datetime(holidays_df['Full_date'])
    holidays_df.rename(columns={'Full_date': 'ds', 'National holiday': 'holiday'}, inplace=True)

    # Load existing models
    model_path = os.path.join(os.getcwd(), 'Models', 'fitted_models_results.pkl')
    with open(model_path, 'rb') as f:
        fitted_models_results = pickle.load(f)

    # Initialize progress bar
    progress_bar = st.progress(0)

    # Train models for each article
    unique_articles = df['Article_name'].unique()
    for idx, article in enumerate(unique_articles):
        try:
            st.write(f"Training model for: {article} ({idx + 1}/{len(unique_articles)})")
            
            # Prepare data for Prophet
            article_data = df[df['Article_name'] == article][['Date', 'Quantity']]
            article_data.columns = ['ds', 'y']

            # Train Prophet model
            model = Prophet(interval_width=0.95, yearly_seasonality=True, holidays=holidays_df)
            model.fit(article_data)

            # Update the fitted_models_results dictionary
            fitted_models_results[article] = {
                'model': model,
                'params': fitted_models_results.get(article, {}).get('params', None),  # Keep existing params if available
                'mae': fitted_models_results.get(article, {}).get('mae', None),  # Keep existing MAE if available
                'accuracy': fitted_models_results.get(article, {}).get('accuracy', None),  # Keep existing accuracy if available
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save the updated models back to the same file
            with open(model_path, 'wb') as f:
                pickle.dump(fitted_models_results, f)

            # Update progress bar
            progress_bar.progress((idx + 1) / len(unique_articles))

        except Exception as e:
            st.warning(f"Failed to train model for {article}: {e}")
            continue

    return fitted_models_results
    
def display_analysis():
    """Displays sales trends for selected categories and articles."""
    if "filtered_data" not in st.session_state:
        st.error("Filtered data is not available. Please ensure categories are mapped.")
        return

    # Retrieve filtered data
    filtered_data = st.session_state.filtered_data.copy()

    # Apply category and article filters
    if selected_categories:
        filtered_data = filtered_data[filtered_data['Category'].isin(selected_categories)]
    if selected_articles:
        filtered_data = filtered_data[filtered_data['Article_name'].isin(selected_articles)]

    # Check if there is any data to display
    if filtered_data.empty:
        st.error("No data available for the selected categories or articles.")
        return

    # Aggregate sales units by date
    daily_sales = filtered_data.groupby('Date')['Quantity'].sum().reset_index()

    # Plot overall daily sales trend
    st.subheader("Overall Daily Sales Trend")
    st.line_chart(daily_sales.set_index('Date'))
    
    # Add day of week column and aggregate sales units by day of week
    filtered_data['Day_of_Week'] = pd.to_datetime(filtered_data['Date'], format='%d/%m/%Y').dt.day_name()
    cumulative_weekly_trend = (
        filtered_data.groupby('Day_of_Week')['Quantity']
        .sum()
        .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        .reset_index()
    )
    weekly_trend = (
        filtered_data.groupby('Day_of_Week')['Quantity']
        .agg(['mean', 'std'])
        .reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        .reset_index()
    )

    # Plot cumulative sales trend by day of the week
    st.subheader("Sales Cumulative Trend by Day of the Week")
    st.bar_chart(cumulative_weekly_trend.set_index('Day_of_Week'))

    # Plot average and standard deviation of sales by day of the week
    st.subheader("Sales Trend by Day of the Week (Mean & Std Dev)")
    st.bar_chart(weekly_trend.set_index('Day_of_Week'))

    # Load models from fitted_models_results.pkl
    with open(model_path, 'rb') as f:
        fitted_models_results = pickle.load(f)

    # Ensure to retrieve the model for each article correctly
    for article in selected_articles:
        st.subheader(f"Analysis and Trends for {article}")

        # Retrieve the model for the article from fitted_models_results
        model_details = fitted_models_results.get(article)
        if model_details is None:
            st.warning(f"No model found for {article}. Skipping...")
            continue
        model = model_details['model']  # access the Prophet model

        # Generate future dataframe for predictions (30 days ahead)
        future = model.make_future_dataframe(periods=30)
        # Get the forecast
        forecast = model.predict(future)

        # Plot general, weekly, and yearly trends using Prophet's built-in plotting functions
        st.subheader(f"General, Weekly, and Yearly Trend for {article}")
        plot1 = model.plot(forecast)
        st.pyplot(plot1)

        # Plot forecast components (e.g., seasonality, trend)
        st.subheader(f"Forecast Components for {article}")
        plot2 = model.plot_components(forecast)
        st.pyplot(plot2)


def generate_sales_projections():
    """
    Generates sales projections for selected articles.
    
    Parameters:
        num_days (int): Number of days to predict ahead.
        
    Returns:
        None: Displays projections in Streamlit.
    """
    if not selected_articles:
        st.warning("Please select at least one article to generate projections.")
    else:
        print(selected_articles) # debugging step
        # Load models from fitted_models_results.pkl
#        model_path = os.path.join(os.getcwd(), 'Models', 'fitted_models_results.pkl')
        with open(model_path, 'rb') as f:
            fitted_models_results = pickle.load(f)

        # Prepare for projections
        num_days = st.number_input('Number of days to predict ahead:', min_value=1,\
                                    max_value=7, value=1,key='num_days_projection')
        projection_date = st.session_state.latest_data_date + timedelta(days=num_days)
        st.write(f"Projection Date: {projection_date.strftime('%A, %d %B %Y')}")

        # Initialize results list
        results = []

        # Process each selected article
        for article in sorted(selected_articles):
            model_details = fitted_models_results.get(article)
            if model_details is None:
                st.warning(f"No model found for {article}. Skipping...")
                continue

            # Access the Prophet model
            prophet_model = model_details['model']

            if article not in fitted_models_results:
                st.warning(f"No model found for {article}. Skipping...")
                continue
            # Get the model from the dictionary
#            model_details = fitted_models_results[article]
#            prophet_model = model_details['model']  # Access the Prophet model
                # Make future predictions
            future = prophet_model.make_future_dataframe(periods=num_days)
            forecast = prophet_model.predict(future)
            predicted_sales = forecast.loc[forecast['ds'] == projection_date, 'yhat'].values
            if len(predicted_sales) == 0:
                st.warning(f"No prediction available for {article} on {projection_date.date()}.")
                continue
            predicted_sales = round(predicted_sales[0], 2)
            # Filter historical data for calculations
            article_data = st.session_state.filtered_data[st.session_state.filtered_data['Article_name'] == article]
            print(article_data.head()) # debugging
            article_data['Date'] = pd.to_datetime(article_data['Date'],format='mixed', dayfirst=True)
            # Calculate averages and medians
            if not article_data.empty:
                past_day_sales = article_data[(article_data['Date'] == projection_date - timedelta(days=1))]['Quantity'].iloc[0]
            else:
                past_day_sales = None  # Or set a default value, e.g., 0
            #past_day_sales = article_data[(article_data['Date'] == projection_date - timedelta(days=1))]['Quantity'].iloc[0] 
            last_7_days = article_data[(article_data['Date'] >= projection_date - timedelta(days=7)) & 
                                        (article_data['Date'] < projection_date)]
            avg_7_days = last_7_days['Quantity'].mean() if not last_7_days.empty else None
            median_7_days = last_7_days['Quantity'].median() if not last_7_days.empty else None
            last_30_days = article_data[(article_data['Date'] >= projection_date - timedelta(days=30)) & 
                                        (article_data['Date'] < projection_date)]
            avg_30_days = last_30_days['Quantity'].mean() if not last_30_days.empty else None
            median_30_days = last_30_days['Quantity'].median() if not last_30_days.empty else None
            date_28_days_from_projection = projection_date - timedelta(days=28)
            sales_same_day_28_days_ago = article_data[(article_data['Date'] == date_28_days_from_projection)]['Quantity'].iloc[0]
            last_year = article_data[(article_data['Date'] >= projection_date - timedelta(days=365)) & 
                            (article_data['Date'] < projection_date)]
            avg_last_year = last_year['Quantity'].mean() if not last_year.empty else None
            median_last_year = last_year['Quantity'].median() if not last_year.empty else None
            same_day_last_year = projection_date - timedelta(days=364)
            sales_same_day_1_year_ago = article_data[article_data['Date']==same_day_last_year]['Quantity'].iloc[0]
            # Append results
            results.append({
                'Projection Date': projection_date.strftime('%A, %d %B %Y'),
                'Category': categories_df[categories_df['Article_name'] == article]['Category'].values[0],
                'Article Name': article,
                'Model Predicted Sales': predicted_sales,
                'Accuracy of the Article Model': fitted_models_results.get(article, {}).get('accuracy', None),  # Keep existing accuracy if available
                'MAPE of the Article Model': fitted_models_results.get(article, {}).get('mape', None),
                'MAE of the Article Model': fitted_models_results.get(article, {}).get('maee', None),
                'Previous Day Sales': past_day_sales,
                'Avg Sales Last 7 Days': avg_7_days,
                'Median Sales Last 7 Days': median_7_days,
                'Avg Sales Last 30 Days': avg_30_days,
                'Median Sales Last 30 Days': median_30_days,
                'Sales Same Day 28 days ago': sales_same_day_28_days_ago,
                'Avg Sales Past Year': avg_last_year,
                'Median Sales Past Year': median_last_year,
                'Sales Same Day One_Year_Ago': sales_same_day_1_year_ago,
                'Decided Quantity for Production': ''
            })

        # Convert results to DataFrame and sort by Category and Article Name
#        results_df = pd.DataFrame(results).sort_values(by=['Category', 'Article Name'])
        results_df = pd.DataFrame(results).sort_values(by='Article Name')
        # Display results
        st.subheader("Sales Projections")
        st.dataframe(results_df)

        # Optionally, allow download of results as Excel file

        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Projections')
        output.seek(0)
        
        st.download_button(
            label="Download Projections as Excel",
            data=output,
            file_name=f'sales_projections_{projection_date.strftime("%Y%m%d")}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

def add_bi_weekly_sales():
    """
    Processes the uploaded bi-weekly sales file and integrates it into historical sales data.
    Parameters:
        uploaded_file: The uploaded CSV file containing bi-weekly sales data.
        historical_data: The existing historical sales data as a DataFrame.
        data_path: The path to save the updated historical sales data.
    Returns:
        None
    """
    uploaded_file=st.file_uploader("Upload a file with recent sales in format such as /narticle Date 1 Date 2/ with respectively the Article_Name and the consecutive quantities sold")
    if uploaded_file is not None:
        try:
            # Step 1: Load bi-weekly sales quantity data
            if uploaded_file.name.endswith('.csv'):
                biweekly_data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xls'):
                biweekly_data = pd.read_excel(uploaded_file)
                logging.info(f"Successfully loaded file: {uploaded_file.name}")
            else:
                logging.error("Unsupported file format uploaded.")
                return
            # Step 2: Aggregate entries by Article_name before processing
            biweekly_data = biweekly_data.groupby("Article", as_index=False).sum()
            logging.info(f"Aggregated bi-weekly data by 'Article'. Rows after aggregation: {len(biweekly_data)}")
            pd.shape(biweekly_data)
            # Step 3: Format bi-weekly data into required format
            formatted_biweekly = pd.melt(
                biweekly_data,
                id_vars=["Article"],
                var_name="Date",
                value_name="Quantity"
            )
            print(formatted_biweekly.head())
            formatted_biweekly['Date'] = pd.to_datetime(formatted_biweekly['Date'], format='%d-%m-%y', dayfirst=True)
            formatted_biweekly['Price'] = float('nan')
            formatted_biweekly.rename(columns={"Article": "Article_name"}, inplace=True)
            logging.info("Formatted bi-weekly data into the required structure.")

            # Step 4: Filter out entries up to the cutoff date (09Jan2025)
            cutoff_date =  datetime.strptime(st.session_state.latest_data_date, "%d%b%Y")
            #datetime.strptime("09Jan2025", "%d%b%Y")
            new_entries = formatted_biweekly[formatted_biweekly['Date'] > cutoff_date]
            logging.info(f"Filtered new entries after cutoff date ({cutoff_date}). Rows after filtering: {len(new_entries)}")

            # Step 5: Merge with historical data to check for duplicates or conflicts
            merged = pd.merge(new_entries, st.session_state.historical_data, on=["Article_name", "Date"], how="left", suffixes=("", "_existing"))
            logging.info("Merged new entries with historical data to check for duplicates and conflicts.")
            # Step 6: Identify duplicates, conflicts, and new rows
            duplicates = merged[(merged["Quantity"] == merged["Quantity_existing"]) &\
                                (merged["Date"] > cutoff_date)]
            print(duplicates)
            conflicts = merged[(merged["Quantity_existing"].notnull()) &\
                            (merged["Quantity"] != merged["Quantity_existing"]) &\
                                (merged["Date"] > cutoff_date)]
            new_rows = merged[(merged["Quantity_existing"].isnull())&\
                            (merged["Date"] > cutoff_date)]
            logging.info(f"Identified {len(duplicates)} duplicates, {len(conflicts)} conflicts, and {len(new_rows)} new rows.")

            # Step 7: Handle conflicts
            rows_to_add = new_rows
            if not conflicts.empty:
                logging.warning(f"Conflicts detected: {conflicts.shape[0]} rows.")
                rows_to_add = pd.concat([new_rows, conflicts])
                logging.info("Conflicting rows will be added.")

            # Step 8: Append validated rows to historical data and save back to CSV
            if not rows_to_add.empty:
                load_historical_data()
                updated_historical_data = pd.concat([st.session_state.historical_data, rows_to_add[["Article_name", "Date", "Price", "Quantity"]]])
                updated_historical_data.to_csv(data_path, index=False, date_format='%d/%m/%Y')                
                logging.info(f"{len(rows_to_add)} new entries successfully added to historical_sales_data.csv.")
                clean_article_names()
                load_historical_data()
                st.session_state.historical_data=st.session_state.historical_data.groupby('Date','Article_Name')[('Price','Quantity')].sum()               
            else:
                logging.info("No new entries were added.")
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")


load_historical_data()
load_category_data()
map_categories_to_historical_data()
load_holidays_date()

print(st.session_state.filtered_data.head())

# Display Latest Model Information
model_paths = [file for file in os.listdir(os.path.join(os.getcwd(), 'Models')) if file.endswith('.pkl')]
if model_paths:
    latest_model_path = max(model_paths, key=lambda x: os.path.getmtime(os.path.join(os.getcwd(), 'Models', x)))
    model_modified_time = datetime.fromtimestamp(os.path.getmtime(os.path.join(os.getcwd(), 'Models', latest_model_path)))
    st.write(f"Latest Model Date: {model_modified_time.date()}")
else:
    st.warning("No models found. Please refresh the model.")

# User Inputs: Select Categories and Articles
categories = categories_df['Category'].dropna().unique()
selected_categories = st.multiselect("Select Categories", categories)

articles = categories_df[categories_df['Category'].isin(selected_categories)]['Article_name'].unique()
selected_articles = st.multiselect("Select Articles", articles, default=articles)

# Action Options: Radio Button for Next Steps
next_action = st.radio(
    "What would you like to do?",
    ["Autocalibrate Model", "Refresh Model", "Display Analysis", "Get Sales Projections","Add Bi-Weekly Sales"], index=None
)

# Handle Radio Button Actions
if next_action == "Autocalibrate Model":
#if 'ds' in filtered_data.columns:
    autocalibrate_prophet(st.session_state.filtered_data, holidays_df, validation_fraction=0.2, seed=42)
#st.error("The 'ds' column is not present in the filtered data. Please ensure the column exists.")
if next_action == "Refresh Model":
    refresh_model(selected_categories=selected_categories, selected_articles=selected_articles)
if next_action == "Display Analysis":
#    display_analysis(filtered_data=st.session_state.filtered_data, selected_categories=selected_categories, selected_articles=selected_articles)
    display_analysis()
if next_action == "Get Sales Projections":
    generate_sales_projections()
elif next_action == "Add Bi-Weekly Sales":
    add_bi_weekly_sales()
    clean_article_names()
    load_historical_data()
    st.session_state.historical_data=st.session_state.historical_data.groupby('Date','Article_Name')               
