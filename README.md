# 🧠 CLTV Prediction with BG-NBD & Gamma-Gamma Models  

This project was completed as part of the Miuul Data Science Bootcamp.  

---

## 📌 Project Overview

**FLO**, a leading online shoe retailer in Turkey, aims to build a roadmap for its future **sales and marketing** strategies. To do so, it needs to estimate the potential value that current customers will bring in the medium and long term.

This project uses **probabilistic modeling techniques** to analyze historical purchasing behaviors and predict the future revenue expected from each customer.

The dataset consists of shopping transactions from customers who made both **online and offline purchases (OmniChannel)** between **2020 and 2021**.


---

## 🗂️ Project Structure

The repository contains the following structure:


├── cltv_prediction.py             # Main Python script containing the CLTV pipeline  
├── requirements.txt               # List of required Python packages  
├── .gitignore                     # Files and folders to ignore in Git  
├── LICENSE                        # Project license  
└── README.md                      # Project documentation  

---

## 📥 Installation

To set up this project on your local machine, follow these steps:


1. Clone the repository:
```
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. (Optional) Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```
---

## ▶️ How to Run

Once the environment is set up, you can run the script with:

```
python cltv_prediction.py
```

⚠️ Note: Ensure that the flo_data_20k_cs2.csv file is located in the same directory as the script, or adjust the file path in the code accordingly.

---

## 📊 Dataset Description

> ℹ️ **Note**: The dataset was provided by **Miuul Bootcamp** and **cannot be shared publicly**.  
> ❗️ **No CSV file is created or included in this repository**; data preprocessing and cleaning are performed directly on the provided dataset.

The dataset contains the following variables:

| Variable                    | Description                                              |
|-----------------------------|----------------------------------------------------------|
| `master_id`                 | Unique customer ID                                       |
| `order_channel`             | Platform used for shopping (Android, iOS, Desktop, Mobile) |
| `last_order_channel`        | Channel of the most recent purchase                      |
| `first_order_date`          | Date of the customer's first purchase                    |
| `last_order_date`           | Date of the customer's last purchase                     |
| `last_order_date_online`    | Date of the last online purchase                         |
| `last_order_date_offline`   | Date of the last offline purchase                        |
| `order_num_total_ever_online` | Total number of online purchases                      |
| `order_num_total_ever_offline` | Total number of offline purchases                    |
| `customer_value_total_ever_online` | Total amount spent on online purchases            |
| `customer_value_total_ever_offline` | Total amount spent on offline purchases          |
| `interested_in_categories_12` | Categories purchased in the last 12 months            |
| `store_type`                | Indicates from which of the 3 company types the customer shopped (e.g., A, B) |

---

## 🔧 Tools and Libraries

- `pandas`, `numpy`: Data manipulation and preprocessing  
- `lifetimes`: Probabilistic CLTV modeling (BG/NBD & Gamma-Gamma)  
- `sklearn`: Feature scaling  
- `datetime`: Date calculations  

---

## 📦 Dependencies

This project relies on the following main Python libraries:

- pandas
- numpy
- scikit-learn
- lifetimes

---

## ⚙️ Project Workflow

1. **Data Cleaning & Preparation**  
   - Outlier treatment using IQR method (capping extreme values)  
   - Combining online & offline purchase metrics into omnichannel variables  
   - Converting date columns to datetime objects for accurate time calculations  

2. **CLTV Feature Engineering**  
   - Calculated Recency, Frequency, T (Tenure), and Monetary values per customer  
   - Used a snapshot date set 2 days after the last purchase in the dataset  

3. **Model Building**  
   - **BG/NBD Model** for estimating expected number of future transactions  
   - **Gamma-Gamma Model** for estimating expected average profit per transaction  
   - Combined models to estimate **6-month CLTV** for each customer  

4. **Customer Segmentation**  
   - Used quantile-based binning (`qcut`) to divide customers into 4 segments: A (top), B, C, D  
   - Generated group-level insights to guide targeted marketing strategies  

---

## 🧠 Insights & Recommendations

### Segment A & B (High CLTV) ⭐
- High purchase frequency and monetary values  
- Recommend loyalty programs, exclusive campaigns, or early-access offers  
- Focus on retention through personalized email or SMS reminders  

### Segment C & D (Low CLTV) 🔄
- Customers may be newer or inactive  
- Offer re-engagement promotions  
- Consider retargeting ads or onboarding journeys to increase engagement  

---

## 📎 Notes

- The project was completed as part of the **Miuul Data Science Bootcamp**.  
- Due to data confidentiality, the dataset is **not available** in this repository.  
- No CSV data file is included or created here; all data processing is done in-code.  
- All modeling and analysis were performed using open-source libraries.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## ✅ Final Remarks

This project demonstrates the power of probabilistic models in customer analytics and shows how businesses can benefit from understanding their customer base in a **data-driven** way. The segmentation derived from CLTV predictions can significantly enhance the efficiency and personalization of marketing efforts.

## 📫 Contact Me

If you have any questions, suggestions, or would like to discuss potential opportunities, feel free to reach out!

---
