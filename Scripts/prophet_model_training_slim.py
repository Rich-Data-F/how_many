# Import dependencies
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../Data/transformed_data/historical_sales_data.csv')
categories_df = pd.read_excel('../Data/transformed_data/categorised_articles.xlsx')

df.head().reset_index(drop=True)

### 3. Apply Data Preprocessing
# Converting Date column to date format
df.Date = pd.to_datetime(df.Date, format="%d/%m/%Y")
df.head()

# Join with categories
df = pd.merge(df, categories_df[['Article_name', 'Category']], on='Article_name', how='left')

# Load holidays data
holidays_df = pd.read_csv('../Data/external_data/holidays.csv')
holidays_df['Full_date'] = pd.to_datetime(holidays_df['Full_date'])
holidays_df = holidays_df.rename(columns={
    'Full_date': 'ds',
    'National holiday': 'holiday'
})

# Train and save models by category
for category in df['Category'].unique():
    fit_models_units = {}
    category_articles = df[df['Category']==category]['Article_name'].unique()
    
    for name in category_articles:
        frame = df[df['Article_name']==name].copy()
        frame.drop(['Article_name', 'Price', 'Category'], axis=1, inplace=True)
        frame.columns = ['ds', 'y']
        m = Prophet(interval_width=0.95, yearly_seasonality=True, holidays=holidays_df)
        model = m.fit(frame)
        fit_models_units[name] = m

    # Save models for this category
    with open(f'../Models/prophet_fit_models_{category}.pkl', 'wb') as f:
        pickle.dump(fit_models_units, f)

# Specify the date you want to get predictions for
target_date = '2025-01-17'  # Change this to your desired date

# Initialize a dictionary to store the predictions
predictions = {}

# Load and predict for each category
for category in df['Category'].unique():
    with open(f'../Models/prophet_fit_models_{category}.pkl', 'rb') as f:
        category_models = pickle.load(f)
        
    for article_name, model in category_models.items():
        # Create a future dataframe for the target date
        future = model.make_future_dataframe(periods=25)
        future['ds'] = pd.to_datetime(future['ds'])
        
        # Get the prediction for the future dates
        forecast = model.predict(future)
        
        # Check if the target date is in the forecast
        if target_date in forecast['ds'].dt.strftime('%Y-%m-%d').values:
            yhat_value = forecast[forecast['ds'] == target_date]['yhat'].values[0]
            predictions[article_name] = yhat_value
        else:
            predictions[article_name] = None

# Convert predictions dictionary to DataFrame
pred_df = pd.DataFrame(list(predictions.items()), columns=['Article_name', 'prediction'])
pred_df = pd.merge(pred_df, categories_df, on='Article_name', how='left')
pred_df['Category'] = pred_df['Category'].fillna('ZZZ_NON_CATEGORISE')
pred_df = pred_df.sort_values(['Category', 'Article_name', 'prediction'], ascending=[True, True, False])
pred_df.to_excel(f'../Data/projections/predicted_units_{target_date}.xlsx', index=False, engine='openpyxl')

# Group by Category and print results
print("Predictions for each article on", target_date)
for category in pred_df['Category'].unique():
    print(f"\n{category}:")
    category_df = pred_df[pred_df['Category'] == category].drop_duplicates('Article_name').sort_values('Article_name')
    for _, row in category_df.iterrows():
        if pd.isna(row['prediction']):
            print(f"  {row['Article_name']}: No prediction available")
        else:
            print(f"  {row['Article_name']}: {row['prediction']:.2f}")

# Update predictions dictionary to maintain sorted order
predictions = dict(zip(pred_df['Article_name'], pred_df['prediction']))

# Create a DataFrame with the predictions
results_df = pd.DataFrame(columns=['Category', 'Article_name', 'Prediction', 'Decision'])

import math

# Populate the DataFrame
for article, yhat in predictions.items():
    if yhat is not None:
        prescription = yhat
        # Round up to next unit if positive, 0 if negative
        decision = max(0, math.ceil(yhat)) if yhat > 0 else 0
    else:
        prescription = "Not available"
        decision = 0
        
    # Get category for this article
    category = categories_df[categories_df['Article_name'] == article]['Category'].iloc[0] \
        if article in categories_df['Article_name'].values else 'ZZZ_NON_CATEGORISE'
        
    results_df = pd.concat([results_df, pd.DataFrame({
        'Category': [category],
        'Article_name': [article],
        'Prediction': [prescription],
        'Decision': [decision]
    })], ignore_index=True)

# Sort the DataFrame and save to Excel
results_df = results_df.sort_values(['Category', 'Article_name', 'Decision'], ascending=[True, True, False])
results_df.to_excel(f'../Data/projections/projections_{target_date}_decision_units.xlsx', index=False, engine='openpyxl')

# Cross validation of models
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Get list of all unique categories and articles
all_categories = results_df['Category'].unique()
selected_categories = random.sample(list(all_categories), min(10, len(all_categories)))
selected_articles = []

# For each category, select one random article
for category in selected_categories:
    category_articles = results_df[results_df['Category'] == category]['Article_name'].unique()
    if len(category_articles) > 0:
        selected_articles.append(random.choice(category_articles.tolist()))

# Create DataFrame to store validation results
validation_results = pd.DataFrame(columns=['Category', 'Article_name', 'MAE', 'RMSE', 'MAPE'])

# Load historical sales data
sales_data = pd.read_csv('../Data/transformed_data/historical_sales_data.csv')
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format='%d/%m/%Y', dayfirst=True)

# For each selected article
for article in selected_articles:
    article_data = sales_data[sales_data['Article_name'] == article].copy()
    
    if len(article_data) > 30:  # Only proceed if we have enough data
        # Split data - keep last 30 days for validation
        cutoff_date = article_data['Date'].max() - pd.Timedelta(days=30)
        train_data = article_data[article_data['Date'] <= cutoff_date]
        test_data = article_data[article_data['Date'] > cutoff_date]
        
        # Prepare training data for Prophet
        train_df = train_data[['Date', 'Quantity']].rename(columns={
            'Date': 'ds',
            'Quantity': 'y'
        })
        
        # Train Prophet model
        model = Prophet()
        model.fit(train_df)
        
        # Make predictions for test period
        future_dates = pd.DataFrame({'ds': test_data['Date']})
        forecast = model.predict(future_dates)
        
        # Calculate error metrics
        y_true = test_data['Quantity'].values
        y_pred = forecast['yhat'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate MAPE avoiding division by zero
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if not np.any(y_true == 0) else np.nan
        
        # Get category for this article
        category = results_df[results_df['Article_name'] == article]['Category'].iloc[0]
        
        # Calculate mean actual and predicted values
        mean_actual = np.mean(y_true)
        mean_predicted = np.mean(y_pred)
        
        # Calculate MAE as percentage of mean actual value
        mae_percentage = (mae / mean_actual * 100) if mean_actual != 0 else np.nan
        
        # Add metrics to validation DataFrame
        validation_results = pd.concat([validation_results, pd.DataFrame({
            'Category': [category],
            'Article_name': [article],
            'MAE': [round(mae, 2)],
            'RMSE': [round(rmse, 2)],
            'MAPE': [round(mape, 2) if not np.isnan(mape) else 'N/A'],
            'Mean_Actual': [round(mean_actual, 2)],
            'Mean_Predicted': [round(mean_predicted, 2)],
            'MAE_Percentage': [round(mae_percentage, 2) if not np.isnan(mae_percentage) else 'N/A']
        })], ignore_index=True)
            
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(test_data['Date'], y_true, label='Actual', marker='o')
        plt.plot(test_data['Date'], y_pred, label='Predicted', marker='o')
        plt.title(f'Actual vs Predicted Values for {article}')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'../Plots/forecast_{article}_{target_date}.png')
        plt.close()

# Sort and display results
validation_results = validation_results.sort_values(['Category', 'Article_name'])
print("\nCross-validation results for randomly selected articles:")
print(validation_results.to_string(index=False))

# Calculate accuracy metrics
validation_results['Accuracy'] = validation_results.apply(
    lambda row: 100 - row['MAPE'] if row['MAPE'] != 'N/A' else 'N/A', 
    axis=1
)

# Format accuracy column to show 2 decimal places
validation_results['Accuracy'] = validation_results['Accuracy'].apply(
    lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
)

# Reorder columns to show accuracy
validation_results = validation_results[['Category', 'Article_name', 'Accuracy', 'MAE', 'RMSE', 'MAPE']]

print("\nUpdated cross-validation results with accuracy:")
print(validation_results.to_string(index=False))

# Save validation results
validation_results.to_excel(f'../Data/projections/cross_validation_results_{target_date}.xlsx', index=False)
