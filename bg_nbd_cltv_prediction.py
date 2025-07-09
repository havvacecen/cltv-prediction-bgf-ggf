# Import necessary libraries
import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

# Set preferred display options for pandas
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 500)         # Increase display width to fit more columns in one line
pd.set_option('display.float_format', lambda x: '%.4f' % x)  # Format floats to 4 decimal places

# TASK 1: Data Preparation

# Step 1: Read the data from "flo_data_20k_cs2.csv"
df_ = pd.read_csv("flo_data_20k_cs2.csv")
df = df_.copy()           # Create a copy of the dataframe to work on
print(df.head())          # Display the first few rows
print(df.shape)           # Check the shape of the dataframe
print(df.isnull().sum())  # Check for missing values in each column
print(df.describe().T)    # Get summary statistics of numeric columns (transposed for readability)

# Step 2: Define functions to handle outliers by capping extreme values
# Note: Frequency values must be integers when calculating CLTV,
# so the upper and lower limits are rounded using round().

def outlier_thresholds(dataframe, variable):
    # Calculate the 5th and 95th percentiles as quartiles
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquartile_range = quartile3 - quartile1
    # Calculate upper and lower limits for outlier capping and round them
    up_limit = round(quartile3 + 1.5 * interquartile_range)
    low_limit = round(quartile1 - 1.5 * interquartile_range)
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    # Get outlier limits
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # Replace values below the lower limit with the lower limit
    dataframe.loc[dataframe[variable] < low_limit, variable] = low_limit
    # Replace values above the upper limit with the upper limit
    dataframe.loc[dataframe[variable] > up_limit, variable] = up_limit

# Step 3: Check and cap outliers in key numeric variables

# Display descriptive statistics for selected variables
print(df[["order_num_total_ever_online",
          "order_num_total_ever_offline",
          "customer_value_total_ever_offline",
          "customer_value_total_ever_online"]].describe().T)

# These variables are likely to contain outliers

# Apply outlier capping to the selected variables
for col in ["order_num_total_ever_online",
            "order_num_total_ever_offline",
            "customer_value_total_ever_offline",
            "customer_value_total_ever_online"]:
    replace_with_thresholds(df, col)

# Display the descriptive statistics again after outlier treatment
print(df[["order_num_total_ever_online",
          "order_num_total_ever_offline",
          "customer_value_total_ever_offline",
          "customer_value_total_ever_online"]].describe().T)

# Step 4: Omnichannel customers shop from both online and offline platforms.
# Create new variables for total number of purchases and total spending per customer.

df["order_num_total_ever_omni"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_ever_omni"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
print(df.head())  # Preview the updated dataframe

# Review statistics for the new variables
print(df[["order_num_total_ever_omni", "customer_value_total_ever_omni"]].describe().T)

# Both new variables seem to contain outliers.
# Let's cap them to prevent issues in the model.

for col in ["order_num_total_ever_omni", "customer_value_total_ever_omni"]:
    replace_with_thresholds(df, col)

print(df.describe().T)  # Check statistics again after capping

# Step 5: Examine variable data types.
# Convert columns containing date information to datetime type.

print(df.dtypes)  # Check current data types

# Identify columns that contain "date" in their name
need_change_cols = [col for col in df.columns if "date" in col]

# Convert these columns to datetime format
df[need_change_cols] = df[need_change_cols].apply(pd.to_datetime)

# Verify the new data types
print(df[need_change_cols].dtypes)

# TASK 2: Creating the CLTV Data Structure

# Step 1: Set the analysis date as 2 days after the most recent transaction date in the dataset

# Get the latest date among all date-related variables
max(df[need_change_cols].max())

# Set the reference date (today_date) for analysis
today_date = max(df[need_change_cols].max()) + pd.Timedelta(days=2)

# Step 2: Create a new CLTV dataframe including:
# customer_id, recency (in weeks), T (tenure in weeks), frequency, and average monetary value per purchase.

print(df.head())  # Preview the dataframe

# Calculate recency: time between the first and last purchase (in weeks)
df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7

# Calculate the customer's 'age' (T) in weeks: time from first purchase to analysis date
df["T_weekly"] = ((today_date - df["first_order_date"]).dt.days) / 7

# Optional sanity check: verify number of unique customers
df["master_id"].nunique()

# Assign total purchase count as frequency
df["frequency"] = df["order_num_total_ever_omni"]

# Calculate average monetary value per purchase
df["monetary_cltv_avg"] = df["customer_value_total_ever_omni"] / df["order_num_total_ever_omni"]

# Create the CLTV dataframe with required variables
cltv_df = df[["master_id", "recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"]]
print(cltv_df.head())  # Preview the resulting CLTV dataframe


# TASK 3: Building BG/NBD and Gamma-Gamma Models & Calculating CLTV.

# Check if there are any negative values in the key columns.
# These models assume non-negative inputs, so it's important to validate.

for col in ["frequency", "recency_cltv_weekly", "T_weekly", "monetary_cltv_avg"]:
    print(f"{col}: {(cltv_df[col] < 0).sum()}")

# Step 1: Fit the BG/NBD model

# Create the model object with a small penalizer coefficient to prevent overfitting
bgf = BetaGeoFitter(penalizer_coef=0.001)

# Fit the model using frequency, recency (in weeks), and T (customer age in weeks)
print(bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"]))

# A) Predict expected number of purchases over the next 3 months (12 weeks)
# Add the results as a new column in the CLTV dataframe
cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(12, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])
print(cltv_df.head())

# B) Predict expected number of purchases over the next 6 months (24 weeks)
# Add the results as another new column in the CLTV dataframe
cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(24, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])
print(cltv_df.head())

# Step 2: Fit the Gamma-Gamma model to estimate the average monetary value per customer

# Create the Gamma-Gamma model object with a small penalizer coefficient to avoid overfitting
ggf = GammaGammaFitter(penalizer_coef=0.001)

# Fit the model using frequency and average monetary value per purchase
print(ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"]))

# Predict the expected average profit (monetary value) for each customer
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
print(cltv_df.head())


# Step 3: Calculate 6-month CLTV and add it to the dataframe as "cltv"

cltv_df["cltv"] = ggf.customer_lifetime_value(
    bgf,
    cltv_df["frequency"],
    cltv_df["recency_cltv_weekly"],
    cltv_df["T_weekly"],
    cltv_df["monetary_cltv_avg"],
    time=6,                # time period in months
    freq="W",              # frequency of the data (weekly)
    discount_rate=0.01)     # monthly discount rate (1%))

# Display the first few rows of the CLTV dataframe
print(cltv_df.head())

# Observe top 20 customers with the highest CLTV
print(cltv_df.sort_values(by="cltv", ascending=False).head(20))

# TASK 4: Creating Segments Based on CLTV Values

# Step 1: Divide all customers into 4 groups (segments) based on their 6-month CLTV and add the segment labels to the dataframe
cltv_df["segment"] = pd.qcut(cltv_df["cltv"], q=4, labels=["D", "C", "B", "A"])
print(cltv_df.head())

# Step 2: Provide brief 6-month action recommendations to management for two selected segments
recommendation_df = cltv_df.groupby("segment").agg({
                                                    "recency_cltv_weekly": "mean",
                                                    "T_weekly": "mean",
                                                    "frequency": "mean",
                                                    "monetary_cltv_avg": "mean",
                                                    "exp_sales_6_month": "mean",
                                                    "cltv": "mean"})
print(recommendation_df)

# Recommendations:
# Customers in segments A and B have high and quite similar expected purchase values.
# Special offers or campaigns can be designed to retain these valuable customers.
# It is observed that the older the first purchase date, the further the customers have drifted from us.
# This suggests a gap in reminding and engaging existing customers.
# Reminder and incentive methods such as emails, SMS campaigns, and special discounts can be applied to reactivate them.
