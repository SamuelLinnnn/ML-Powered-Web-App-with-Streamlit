import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Load data for visualization
data = pd.read_csv("C:/Users/Samuel Lin/Downloads/ML-Powered Web App/Python_Web_App_Feature1&2/shopping_behavior_updated.csv")
# Set page config
st.set_page_config(
    page_title="Market Analysis Dashboard",
    page_icon="📊",
    initial_sidebar_state="expanded"
)
# Sidebar for feature selection
st.sidebar.title("Menu")
menu = st.sidebar.selectbox(
    "Select Feature",
    [
        "Home",
        "Market Position Analysis",
        "Competitive Analysis",
        "ML Model Application",
    ],
)


# Front Page
def display_home():
    """
    Display the home section of the web application.

    This function uses Streamlit to:
      - Show a title indicating the app's analytical features.
      - Provide an overview of how machine learning can help
        businesses understand market position, analyze competition,
        and identify growth opportunities.

    Usage:
      - Call this function at the start of your Streamlit app
        to welcome users and highlight the app’s purpose.

    Returns:
      None
    """
    st.title("Web App with Analytical Features")
    st.markdown(
        """
        Welcome to the Machine Learning Web App! This application leverages powerful machine learning models to analyze market data and provide actionable insights. 

        The goal is to help businesses:

        - Understand their market position
        - Analyze competition
        - Identify growth opportunities
        - Develop data-driven strategies

        Explore the features to gain valuable insights for your business success.
        """
    )



# Market Position Analysis
def market_position_analysis(selected_graph):
    """
    Display a selected market analysis chart and related insights in the Streamlit app.

    Parameters:
    -----------
    selected_graph : str
        The name of the chart or analysis to display. Possible options include:
        - "Market Share by Category"
        - "Average Price by Category"
        - "Sales by Season"
        - "Average Ratings by Category"
        - "Average Ratings by Category (Pie)"
        - "Average Ratings by Category (Bar)"
        - "Category Sales by Gender"
        - "Most Purchased Cloth Size by Season"
        - "Market Share by Category (Pie)"
        - "Category Ranking by Season"
        - "Cloth Purchase Trend by Gender and Season"

    Usage:
    ------
    market_position_analysis("Market Share by Category")

    Return: Does not return any value; it directly renders visualizations 
    and text output within a Streamlit interface.
    """

    # If the user selects "Market Share by Category," display a pie chart 
    # showing how each category contributes to total sales.
    if selected_graph == "Market Share by Category":
        st.subheader("Market Share by Category")
        category_share = data.groupby("Category")["Purchase Amount (USD)"].sum()
        fig, ax = plt.subplots()
        category_share.plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax)
        ax.set_ylabel("")
        ax.set_title("Market Share by Category")
        st.pyplot(fig)

        # Provide insights about the chart.
        st.write("The 'Market Share by Category' visualization illustrates the proportion of total sales contributed by each category.")
        st.write("Clothing holds the largest share at 44.7%, indicating its dominance in the market.")
        st.write("Accessories follow with a 31.8% share, reflecting significant customer interest in this category.")
        st.write("Footwear accounts for 15.5% of the market, while Outerwear represents the smallest portion at 7.9%.")
        st.write("This breakdown highlights potential growth opportunities for categories with lower market shares.")
        st.write("Businesses can focus on targeted promotions and product diversification to improve the market presence of Footwear and Outerwear.")

    # If the user selects "Average Price by Category," display a bar chart 
    # showing the mean purchase amount for each category.
    elif selected_graph == "Average Price by Category":
        st.subheader("Average Price by Category")
        avg_price = data.groupby("Category")["Purchase Amount (USD)"].mean()
        fig, ax = plt.subplots()
        avg_price.plot(kind="bar", ax=ax)
        ax.set_ylabel("Average Price (USD)")
        ax.set_title("Average Price by Category")
        ax.set_ylim([50, 70])
    # Add values on top of each bar
        for bar in ax.patches:
            ax.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate (center of bar)
            bar.get_height() + (avg_price.max() * 0.02),  # Slight offset above bar
            f"{bar.get_height():.2f}",  # Format to 2 decimal places
            ha="center",  # Horizontal alignment
            va="bottom"  # Vertical alignment
        )
        st.pyplot(fig)
        
        st.write("The Average Price by Category visualization highlights that Footwear has the highest average price at $60.26.")
        #st.write("Then it follows by Clothing at $60.03      and    Accessories   at    $59.84.")
        st.write("Outerwear stands out with the lowest average price at $57.17.")
        st.write("")
        st.write("This indicates that Outerwear products may be priced more competitively or may include lower-priced items.")
        st.write("Conversely, Footwear and Clothing likely have premium products contributing to their higher averages.")
        st.write("This trend provides insights for pricing strategies and may influence promotional offers or product bundling.")


    # For "Sales by Season," display a bar chart showing total sales per season.
    elif selected_graph == "Sales by Season":
        st.subheader("Sales by Season")
        growth = data.groupby("Season")["Purchase Amount (USD)"].sum()
        fig, ax = plt.subplots()
        growth.plot(kind="bar", ax=ax)
        ax.set_ylabel("Total Sales (USD)")
        ax.set_title("Sales by Season")
        st.pyplot(fig)
        
        st.write("The 'Sales by Season' visualization highlights fluctuations in total sales across different seasons.")
        st.write("Fall has the highest sales, reaching over 60,000 USD, indicating a peak shopping period.")
        st.write("Winter also shows strong sales, likely due to holiday purchases.")
        st.write("Spring and Summer have comparatively lower sales, with Summer being the lowest at around 53,000 USD.")
        st.write("These insights suggest that businesses can optimize inventory and marketing strategies around Fall and Winter to maximize revenue.")
        st.write("For Spring and Summer, targeted promotions or new product launches may help boost sales.")

    # Displays a bar chart of average ratings by category.
    elif selected_graph == "Average Ratings by Category":
        st.subheader("Average Ratings by Category")
        avg_rating = data.groupby("Category")["Review Rating"].mean()
        fig, ax = plt.subplots()
        avg_rating.plot(kind="bar", color="orange", ax=ax)
        ax.set_ylabel("Average Rating")
        ax.set_title("Average Ratings by Category")
        plt.xticks(rotation=45)
        ax.set_ylim([3.0, 5.0])

        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (avg_rating.max() * 0.02),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom"
            )
        st.pyplot(fig)
        st.write("The 'Average Ratings by Category' visualization shows that customer ratings are fairly consistent across categories.")
        st.write("Footwear has the highest average rating at 3.79, indicating positive feedback for this category.")
        st.write("Accessories and Outerwear follow closely with average ratings of 3.77 and 3.75, respectively.")
        st.write("Clothing has the lowest average rating at 3.72, though the difference is minimal.")
        st.write("These ratings suggest a generally favorable perception of the products, with little variation between categories.")
        st.write("This can help the business maintain high-quality standards across categories and prioritize improvements where needed.")

            
    elif selected_graph == "Average Ratings by Category (Pie)":
        st.subheader("Average Ratings by Category (Pie)")
        avg_rating = data.groupby("Category")["Review Rating"].mean()
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            avg_rating,
            labels=avg_rating.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=sns.color_palette("bright"),
        )
        ax.set_title("Average Ratings by Category (Pie)")
        plt.tight_layout()
        st.pyplot(fig)
        st.write("The 'Average Ratings by Category' pie chart shows a relatively balanced distribution of ratings across all categories.")
        st.write("Footwear has the highest proportion at 25.2%, indicating slightly better customer satisfaction compared to other categories.")
        st.write("Accessories follow closely at 25.1%, while Clothing and Outerwear contribute 24.8% and 24.9%, respectively.")
        st.write("The small differences suggest consistent product quality and customer experience across categories.")
        st.write("Businesses should maintain this balance while identifying specific areas for improvement to boost overall ratings.")


    elif selected_graph == "Average Ratings by Category (Bar)":
        st.subheader("Average Ratings by Category (Bar)")
        avg_rating = data.groupby("Category")["Review Rating"].mean()
        fig, ax = plt.subplots()
        avg_rating.plot(kind="bar", color="orange", ax=ax)
        ax.set_ylabel("Average Rating")
        ax.set_title("Average Ratings by Category")
        plt.xticks(rotation=45)
        ax.set_ylim([3.0, 5.0])
        plt.tight_layout()

        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (avg_rating.max() * 0.02),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom"
            )
        st.pyplot(fig)
        st.write("The 'Average Ratings by Category' visualization shows that customer ratings are fairly consistent across categories.")
        st.write("Footwear has the highest average rating at 3.79, indicating positive feedback for this category.")
        st.write("Accessories and Outerwear follow closely with average ratings of 3.77 and 3.75, respectively.")
        st.write("Clothing has the lowest average rating at 3.72, though the difference is minimal.")
        st.write("These ratings suggest a generally favorable perception of the products, with little variation between categories.")
        st.write("This can help the business maintain high-quality standards across categories and prioritize improvements where needed.")



    elif selected_graph == "Category Sales by Gender":
        st.subheader("Category Sales by Gender")
        gender_category_sales = (
            data.groupby(["Gender", "Category"])["Purchase Amount (USD)"].sum().unstack()
        )
        fig, ax = plt.subplots()
        gender_category_sales.plot(
            kind="bar", stacked=True, ax=ax, color=sns.color_palette("bright")
        )
        ax.set_ylabel("Total Sales (USD)")
        ax.set_title("Category Sales by Gender")
        ax.set_xlabel("Gender")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        st.write("The 'Category Sales by Gender' visualization compares the total sales for each category across male and female customers.")
        st.write("Sales generated by male customers significantly surpass those by female customers, indicating a larger male consumer base or higher spending.")
        st.write("Clothing forms the largest portion of sales for both genders, followed by Accessories.")
        st.write("Footwear and Outerwear make up smaller proportions, suggesting potential for further promotion within these categories.")
        st.write("Businesses could tailor gender-specific marketing strategies to maximize sales and address underperforming categories.")


    elif selected_graph == "Most Purchased Cloth Size by Season":
        st.subheader("Most Purchased Cloth Size by Season")
        season_size_sales = (
            data.groupby(["Season", "Size"])["Purchase Amount (USD)"].sum().unstack()
        )
        fig, ax = plt.subplots()
        season_size_sales.plot(
            kind="bar", stacked=True, ax=ax, color=sns.color_palette("bright")
        )
        ax.set_ylabel("Total Sales (USD)")
        ax.set_title("Most Purchased Cloth Size by Season")
        ax.set_xlabel("Season")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        st.write("The 'Most Purchased Cloth Size by Season' visualization indicates that size L consistently accounts for the highest sales across all seasons.")
        st.write("Size M also shows significant sales, although not as high as size L, indicating it is a popular choice.")
        st.write("Sizes S and XL have comparatively lower sales, reflecting either lower demand or limited availability.")
        st.write("")
        st.write("The consistent trend across all seasons suggests stable consumer preferences for larger sizes.")
        st.write("Businesses can use this insight to optimize inventory for sizes L and M, ensuring they meet seasonal demand.")
        st.write("Targeted promotions for smaller sizes may also help boost sales in underperforming categories.")
        

    elif selected_graph == "Market Share by Category (Pie)":
        st.subheader("Market Share by Category (Pie)")
        category_share = data.groupby("Category")["Purchase Amount (USD)"].sum()
        fig, ax = plt.subplots()
        category_share.plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax)
        ax.set_ylabel("")
        ax.set_title("Market Share by Category")
        st.pyplot(fig)

    elif selected_graph == "Category Ranking by Season":
        st.subheader("Category Ranking by Season")
        season_category_sales = (
            data.groupby(["Season", "Category"])["Purchase Amount (USD)"]
            .sum()
            .reset_index()
        )
        fig, ax = plt.subplots()
        sns.barplot(
            data=season_category_sales,
            x="Season",
            y="Purchase Amount (USD)",
            hue="Category",
            ax=ax,
            palette="bright",
        )
        ax.set_ylabel("Total Sales (USD)")
        ax.set_title("Category Ranking by Season")
        plt.tight_layout()
        st.pyplot(fig)
        st.write("The 'Category Ranking by Season' visualization showcases sales distribution across categories for each season.")
        st.write("Clothing consistently leads across all seasons, indicating strong consumer demand regardless of the time of year.")
        st.write("Accessories follow as the second highest, showing steady performance, particularly during Fall and Winter.")
        st.write("Footwear shows moderate sales, peaking in Winter, which aligns with seasonal demand for boots and warmer footwear.")
        st.write("Outerwear sales are notably lower overall, peaking in Winter as expected.")
        st.write("This data suggests focusing on Clothing and Accessories for year-round inventory and promotions, while emphasizing Outerwear during Winter.")


    elif selected_graph == "Cloth Purchase Trend by Gender and Season":
        st.subheader("Cloth Purchase Trend by Gender and Season")
        trend_data = (
            data[data["Category"] == "Clothing"]
            .groupby(["Season", "Gender"])["Purchase Amount (USD)"]
            .sum()
            .reset_index()
        )
        trend_data = trend_data.pivot(
            index="Season", columns="Gender", values="Purchase Amount (USD)"
        ).reset_index()
        fig, ax = plt.subplots()
        trend_data.plot(x="Season", marker="o", ax=ax)
        ax.set_ylabel("Total Sales (USD)")
        ax.set_title("Cloth Purchase Trend by Gender and Season")
        plt.tight_layout()
        st.pyplot(fig)
        st.write("The 'Cloth Purchase Trend by Gender and Season' visualization illustrates seasonal purchasing behavior across genders.")
        st.write("Male customers show significant fluctuations, with a peak in Spring followed by a dip in Summer, and another rise in Winter.")
        st.write("Female customers' purchasing trend is relatively stable, with a slight increase in Winter.")
        st.write("This disparity suggests that males may respond more strongly to seasonal changes or promotions, particularly during colder seasons.")
        st.write("Businesses can leverage this trend by tailoring seasonal marketing campaigns to male consumers, especially during Winter, while maintaining steady promotions for female customers.")


# Competitive Analysis
def competitive_analysis(selected_graph):
    """
    Display a selected competitive analysis chart and corresponding insights.

    Args:
        selected_graph (str):
            A string indicating which chart or analysis to display. Options include:
            - "Total Purchase Amount by Age Group, Gender, and Shipping Type"
            - "Box Plot of Purchase Amounts"
            - "Top 10 Most Purchased Products"
            - "Least 10 Most Purchased Products"
            - "Popular Payment Methods"
            - "Revenue by Product Category"
            - "Preferred Shipping Options by Age Group"
            - "Sales Volume Across Price Ranges"
            - "Purchase Frequency by Gender"
            - "Purchase Frequency by Age Group and Gender"
    
    Usage:
        competitive_analysis("Top 10 Most Purchased Products")

    Return: does not return any value; instead, it renders the chart
    and text explanations directly within the Streamlit interface. 
    """

    if selected_graph == "Total Purchase Amount by Age Group, Gender, and Shipping Type":
        st.subheader("Total Purchase Amount by Age Group, Gender, and Shipping Type")
        age_bins = [0, 18, 25, 35, 45, 55, 65, 100]
        age_labels = ["0-18", "19-25", "26-35", "36-45", "46-55", "56-65", "66+"]
        data["Age Group"] = pd.cut(
            data["Age"], bins=age_bins, labels=age_labels, right=False
        )
        agg_data = (
            data.groupby(["Age Group", "Gender", "Shipping Type"])["Purchase Amount (USD)"]
            .sum()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            data=agg_data,
            x="Age Group",
            y="Purchase Amount (USD)",
            hue="Gender",
            errorbar=None,
            ax=ax,
        )
        ax.set_title("Total Purchase Amount by Age Group, Gender, and Shipping Type")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Total Purchase Amount (USD)")
        ax.legend(title="Gender")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        st.write("The visualization shows the total purchase amount segmented by age group, gender, and shipping type.")
        st.write("Male customers consistently spend more than female customers across all age groups, with the 26–35 and 46–55 age groups showing the highest expenditures.")
        st.write("Female spending peaks in the 26–35 age group but remains significantly lower than male spending.")
        st.write("The 0–18 and 66+ age groups have the lowest total purchase amounts for both genders.")
        st.write("This indicates that marketing strategies should focus on male consumers in their peak spending age groups, while campaigns targeting women could emphasize categories appealing to the 26–35 age group.")


    elif selected_graph == "Box Plot of Purchase Amounts":
        st.subheader("Box Plot of Purchase Amounts")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=data["Purchase Amount (USD)"], color="lightcoral", ax=ax)
        ax.set_title("Box Plot of Purchase Amounts")
        ax.set_xlabel("Purchase Amount (USD)")
        st.pyplot(fig)
        st.write("The box plot provides an overview of the distribution of purchase amounts.")
        st.write("The median purchase amount appears close to the center of the interquartile range (IQR), suggesting a relatively balanced distribution.")
        st.write("The IQR, which spans from around $40 to $80, represents the middle 50% of purchases.")
        st.write("The whiskers indicate that most purchases fall between approximately $20 and $100, with minimal outliers.")
        st.write("This analysis shows that while most transactions are clustered within a mid-range value, the data contains no extreme anomalies.")

    # Scatter plot (box plot version since I tried scatterplot and I found the plot is terrible)
    elif selected_graph == 'Price Points vs. Review Ratings (Binned Price Ranges)':
        st.subheader("Box Plot: Price Points vs. Review Ratings with Binned Prices (Originally it should be scatterplot)")

    # Create price bins
        data["Price Bin"] = pd.cut(
            data["Purchase Amount (USD)"],
            bins=[0, 20, 40, 60, 80, 100],
            labels=["0-20", "21-40", "41-60", "61-80", "81-100"]
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=data,
            x="Price Bin",
            y="Review Rating",
            hue="Category",
            palette="Set2",
            ax=ax
        )
        ax.set_title("Price Points vs. Review Ratings (Binned Price Ranges)")
        ax.set_xlabel("Price Range (USD)")
        ax.set_ylabel("Review Rating")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        st.pyplot(fig)
        st.write("The box plot visualization shows the distribution of review ratings for products within different price ranges. Higher-priced products generally exhibit a wider range of review ratings, indicating varied customer satisfaction.")
        st.write("Across all price ranges, the median review ratings remain consistently high, hovering around 4 out of 5, suggesting that most products, regardless of price, are well-received.")
        st.write("However, the interquartile ranges reveal some differences in consistency, with some categories showing more variation in customer feedback at lower price points.")
        st.write("Footwear and accessories display slightly higher median ratings compared to outerwear and clothing, especially in higher price ranges.")
        st.write("**Additional Note**: I tried the scatter plot and it appears overcrowded due to the large dataset, making it difficult to identify trends as individual points overlap heavily.")
        st.write("To improve clarity, a binned box plot was used to group data points by price ranges, providing a clearer summary of review rating distributions across price tiers.")
    
    elif selected_graph == "Top 10 Most Purchased Products":
        st.subheader("Top 10 Most Purchased Products")
        product_counts = data["Item Purchased"].value_counts()
        top_10_products = product_counts.head(10)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x=top_10_products.values, y=top_10_products.index, palette="viridis", ax=ax
        )
        ax.set_title("Top 10 Most Purchased Products")
        ax.set_xlabel("Number of Purchases")
        ax.set_ylabel("Product")
        for i, value in enumerate(top_10_products.values):
            ax.text(
                value + 5, i, f"{value}", va="center", ha="left", color="black", fontsize=12
            )
        plt.tight_layout()
        st.pyplot(fig)

    elif selected_graph == "Least 10 Most Purchased Products":
        st.subheader("Least 10 Most Purchased Products")
        product_counts = data["Item Purchased"].value_counts()
        bottom_10_products = product_counts.tail(10)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            x=bottom_10_products.values, y=bottom_10_products.index, palette="viridis", ax=ax
        )
        ax.set_title("Least 10 Most Purchased Products")
        ax.set_xlabel("Number of Purchases")
        ax.set_ylabel("Product")
        for i, value in enumerate(bottom_10_products.values):
            ax.text(
                value + 5, i, f"{value}", va="center", ha="left", color="black", fontsize=12
            )
        plt.tight_layout()
        st.pyplot(fig)
        st.write("The Top 10 Most Purchased Products reveal that Blouse, Pants, and Jewelry top the list with 171 purchases each, indicating high demand for these categories.")
        st.write("Shirts, Dresses, and Sweaters also appear prominently, reflecting a preference for versatile clothing items.")
        st.write("Conversely, the Least 10 Most Purchased Products show that Hats and Handbags have lower demand, with Hats leading at 154 purchases and Jeans at the bottom with 124 purchases.")
        st.write("This comparison highlights that fashion accessories like Blouses and Jewelry are highly favored, while basics like Jeans show comparatively less interest.")
        st.write("Businesses can use this insight to adjust inventory levels and plan marketing campaigns tailored to popular and underperforming items.")
        

    elif selected_graph == "Popular Payment Methods":
        st.subheader("Popular Payment Methods")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            y="Payment Method",
            data=data,
            palette="pastel",
            order=data["Payment Method"].value_counts().index,
            ax=ax,
        )
        ax.set_title("Popular Payment Methods")
        ax.set_xlabel("Number of Purchases")
        ax.set_ylabel("Payment Method")
        max_value = data["Payment Method"].value_counts().max()
        ax.set_xlim(0, max_value + 50)
        for i, value in enumerate(data["Payment Method"].value_counts().values):
            ax.text(
                value + 5, i, f"{value}", va="center", ha="left", color="black", fontsize=12
            )
        st.pyplot(fig)
        st.write("This bar chart illustrates the distribution of popular payment methods among customers.")
        st.write("PayPal is the most frequently used payment method with 677 purchases, closely followed by credit card (671) and cash (670).")
        st.write("Debit cards and Venmo are moderately popular, accounting for 636 and 634 purchases respectively.")
        st.write("Bank transfers are the least preferred method, with 612 purchases.")
        st.write("The data suggests that customers prefer digital payment methods such as PayPal and credit cards over traditional methods like bank transfers.")
        st.write("Businesses could consider incentivizing lesser-used payment options to diversify customer preferences and reduce transaction fees.")


    elif selected_graph == "Revenue by Product Category":
        st.subheader("Revenue by Product Category")
        category_revenue = (
            data.groupby("Category")["Purchase Amount (USD)"].sum().reset_index()
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            x="Category",
            y="Purchase Amount (USD)",
            data=category_revenue,
            palette="muted",
            ax=ax,
        )
        ax.set_title("Revenue by Product Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Total Revenue (USD)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write("This bar chart illustrates the total revenue generated by different product categories.")
        st.write("Clothing is the highest revenue contributor, surpassing $100,000, indicating a strong demand for apparel.")
        st.write("Accessories follow as the second-largest revenue generator, contributing approximately $70,000.")
        st.write("Footwear and Outerwear generate significantly lower revenues compared to Clothing and Accessories.")
        st.write("These insights suggest that focusing promotional efforts and inventory optimization on Clothing and Accessories may yield the highest returns.")


    elif selected_graph == "Preferred Shipping Options by Age Group":
        age_bins = [18, 25, 35, 45, 55, 65, 75]
        age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74']
        data['Age Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)
        
        st.subheader("Preferred Shipping Options by Age Group")
        if "Age Group" not in data.columns:
            st.error("The 'Age Group' column is missing in the data. Please check the input data.")
            return
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.countplot(
            x="Age Group", hue="Shipping Type", data=data, palette="viridis", ax=ax
        )
        ax.set_title("Preferred Shipping Options by Age Group")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Number of Purchases")
        ax.legend(title="Shipping Type")
        st.pyplot(fig)
        st.write("This bar chart displays the distribution of preferred shipping options across various age groups.")
        st.write("The 26-35 age group shows the highest number of purchases, with a noticeable preference for free shipping and standard shipping.")
        st.write("Express and next-day air shipping options are less commonly chosen across all age groups, indicating that speed may not be the top priority for most consumers.")
        st.write("The 0-18 and 66+ age groups demonstrate lower overall purchase activity and a preference for store pickup and free shipping.")
        st.write("This data suggests that optimizing free and standard shipping options could enhance customer satisfaction, especially among younger and middle-aged shoppers.")


    elif selected_graph == "Sales Volume Across Price Ranges":
        st.subheader("Sales Volume Across Price Ranges")
        price_bins = [0, 50, 100, 150, 200, 250, 300]
        price_labels = ["$0-50", "$51-100", "$101-150", "$151-200", "$201-250", "$251-300"]
        data["Price Range"] = pd.cut(
            data["Purchase Amount (USD)"], bins=price_bins, labels=price_labels, right=False
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x="Price Range", data=data, palette="Spectral", ax=ax)
        ax.set_title("Sales Volume Across Price Ranges")
        ax.set_xlabel("Price Range (USD)")
        ax.set_ylabel("Number of Purchases")
        st.pyplot(fig)
        st.write("This bar chart illustrates the sales volume across different price ranges.")
        st.write("The $51–100 price range dominates the sales volume, indicating that this range is the most preferred by consumers.")
        st.write("The $0–50 range also shows strong sales, reflecting a significant demand for lower-priced products.")
        st.write("In contrast, sales drop significantly for price ranges above $100, indicating that higher-priced products are purchased less frequently.")
        st.write("This suggests a consumer preference for affordable to moderately priced items. Businesses may benefit from focusing promotions and product bundles within these ranges to boost revenue.")


    elif selected_graph == "Purchase Frequency by Gender":
        st.subheader("Purchase Frequency by Gender")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x="Gender", data=data, palette="Set2", ax=ax)
        ax.set_title("Purchase Frequency by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of Purchases")
        st.pyplot(fig)
        st.write("This bar chart shows the purchase frequency by gender.")
        st.write("Male customers have a significantly higher number of purchases compared to female customers.")
        st.write("This trend suggests that male consumers may be more active shoppers in this dataset or could indicate a higher level of engagement from male customers.")
        st.write("Understanding the reasons behind this gender-based difference can help in tailoring marketing strategies to encourage higher participation from female customers.")
        st.write("This insight can guide businesses to implement gender-specific promotions or loyalty programs to balance the customer base.")


    elif selected_graph == "Purchase Frequency by Age Group and Gender":
        st.subheader("Purchase Frequency by Age Group and Gender")
        if "Frequency of Purchases" not in data.columns:
            st.error("The 'Frequency of Purchases' column is missing in the data. Please check the input data.")
            return

        age_bins = [18, 25, 35, 45, 55, 65, 75]
        age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74']
        data['Age Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=False)

        freq_demo = (
            data.groupby(["Age Group", "Gender"])["Frequency of Purchases"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )

        fig, ax = plt.subplots(figsize=(14, 8))
        freq_demo.plot(kind="bar", stacked=True, colormap="Paired", ax=ax)
        ax.set_title("Purchase Frequency by Age Group and Gender")
        ax.set_xlabel("Age Group and Gender")
        ax.set_ylabel("Proportion of Purchase Frequency")
        ax.legend(title="Purchase Frequency", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)
        st.write("This stacked bar chart depicts the purchase frequency distribution across age groups and gender.")
        st.write("Weekly purchases are the most common frequency across all age groups and genders, indicating consistent shopping habits.")
        st.write("Bi-weekly and monthly purchases also contribute significantly, particularly in middle-aged and younger groups.")
        st.write("Annual and quarterly purchases form a small proportion across all categories, suggesting that long-term purchases are rare.")
        st.write("Interestingly, the 25-35 and 36-45 male groups exhibit a more even distribution across frequent purchase categories compared to other groups.")
        st.write("This data can help in designing age-specific promotions to maximize sales by targeting the most frequent purchase cycles.")



# ML Model Functions (Regression and Classification)
# regression_model 
def regression_model():
    """
    Build and display a linear regression model for predicting purchase amounts.

    Usage:
        regression_model()

    Requirements:
      - A global 'data' DataFrame with columns:
        ["Age", "Gender", "Category", "Location", "Season",
         "Subscription Status", "Previous Purchases", "Purchase Amount (USD)"]
      - Required libraries imported (Streamlit, numpy, pandas, sklearn, etc.).
    """
    st.title("Regression: Predict Purchase Amount")

    # Prepare data
    features = data[
        [
            "Age",
            "Gender",
            "Category",
            "Location",
            "Season",
            "Subscription Status",
            "Previous Purchases",
        ]
    ]
    target = data["Purchase Amount (USD)"]

    categorical_features_reg = [
        "Gender",
        "Category",
        "Location",
        "Season",
        "Subscription Status",
    ]
    encoder_reg = OneHotEncoder(drop="first", sparse_output=False)
    scaler_reg = StandardScaler()

    encoded_features = encoder_reg.fit_transform(features[categorical_features_reg])
    numerical_features = features[["Age", "Previous Purchases"]].values
    X = np.hstack((numerical_features, encoded_features))
    X_scaled = scaler_reg.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, target, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Input for prediction
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    category = st.selectbox("Category", np.unique(data["Category"].values).tolist())
    location = st.selectbox(
        "location", np.unique(data["Location"].values).tolist()
    )
    season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
    subscription_status = st.selectbox("Subscription Status", ["Yes", "No"])
    previous_purchases = st.number_input(
        "Previous Purchases", min_value=0, max_value=100, value=5
    )

    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Category": [category],
            "Location": [location],
            "Season": [season],
            "Subscription Status": [subscription_status],
            "Previous Purchases": [previous_purchases],
        }
    )

    # Preprocess input
    encoded_input = encoder_reg.transform(input_data[categorical_features_reg])
    combined_input = np.hstack(
        (input_data[["Age", "Previous Purchases"]].values, encoded_input)
    )
    scaled_input = scaler_reg.transform(combined_input)

    if st.button("Predict Purchase Amount"):
        prediction = model.predict(scaled_input)
        st.write(f"### Predicted Purchase Amount: ${prediction[0]:.2f}")


# classification_model 
def classification_model():
    """
    Build and display a classification model to predict subscription status.

    Usage:
        classification_model()

    Requirements:
        - A global 'data' DataFrame with necessary columns:
            "Age", "Gender", "Purchase Amount (USD)", "Location",
            "Season", "Previous Purchases", "Frequency of Purchases",
            "Subscription Status".
        - Libraries: Streamlit, pandas, numpy, sklearn, etc.
    """

    st.title("Classification: Predict Subscription Status")
    # Prepare data
    features = data[
        [
            "Age",
            "Gender",
            "Purchase Amount (USD)",
            "Location",
            "Season",
            "Previous Purchases",
            "Frequency of Purchases",
        ]
    ]
    target = data["Subscription Status"].apply(lambda x: 1 if x == "Yes" else 0)

    categorical_features_clf = [
        "Gender",
        "Location",
        "Season",
        "Frequency of Purchases",
    ]
    encoder_clf = OneHotEncoder(drop="first", sparse_output=False)
    scaler_clf = StandardScaler()

    encoded_features = encoder_clf.fit_transform(features[categorical_features_clf])
    numerical_features = features[
        ["Age", "Purchase Amount (USD)", "Previous Purchases"]
    ].values
    X = np.hstack((numerical_features, encoded_features))
    X_scaled = scaler_clf.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, target, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    

    # Input for prediction
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    purchase_amount = st.number_input(
        "Purchase Amount (USD)", min_value=0.0, value=50.0
    )
    # location = st.text_input("Location", "Maine")
    location = st.selectbox(
        "location", np.unique(data["Location"].values).tolist()
    )
    season = st.selectbox("Season", np.unique(data["Season"].values).tolist())
    previous_purchases = st.number_input(
        "Previous Purchases", min_value=0, max_value=100, value=5
    )
    frequency_of_purchases = st.selectbox(
        "Frequency of Purchases", np.unique(data["Frequency of Purchases"].values).tolist()
    )

    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Purchase Amount (USD)": [purchase_amount],
            "Location": [location],
            "Season": [season],
            "Previous Purchases": [previous_purchases],
            "Frequency of Purchases": [frequency_of_purchases],
        }
    )

    # Preprocess input
    encoded_input = encoder_clf.transform(input_data[categorical_features_clf])
    combined_input = np.hstack(
        (
            input_data[["Age", "Purchase Amount (USD)", "Previous Purchases"]].values,
            encoded_input,
        )
    )
    scaled_input = scaler_clf.transform(combined_input)

    if st.button("Predict Subscription Status"):
        prediction = model.predict(scaled_input)
        status = "Subscribed" if prediction[0] == 1 else "Not Subscribed"
        st.write(f"### Predicted Subscription Status: {status}")


# Display based on menu selection
if menu == "Home":
    display_home()
elif menu == "Market Position Analysis":
    st.header("Market Position Analysis")
    graph_options = [
        "Sales by Season",
        "Category Sales by Gender",
        "Market Share by Category",
        "Average Price by Category",
        "Category Ranking by Season",
        "Average Ratings by Category (Pie)",
        "Average Ratings by Category (Bar)",
        "Most Purchased Cloth Size by Season",
        "Cloth Purchase Trend by Gender and Season",
        # Add other Market Position Analysis options here
    ]
    selected_graph = st.selectbox("Choose a graph to display:", ["Market Position Analysis Graphs"] + graph_options)
    if selected_graph:
        market_position_analysis(selected_graph)
elif menu == "Competitive Analysis":
    st.header("Competitive Analysis")
    graph_options = [
        "Popular Payment Methods",
        "Revenue by Product Category",
        "Box Plot of Purchase Amounts",
        "Price Points vs. Review Ratings (Binned Price Ranges)",
        "Purchase Frequency by Gender",
        "Top 10 Most Purchased Products",
        "Least 10 Most Purchased Products",
        "Sales Volume Across Price Ranges",
        "Preferred Shipping Options by Age Group",
        "Purchase Frequency by Age Group and Gender",
        "Total Purchase Amount by Age Group, Gender, and Shipping Type",
        # Add other Competitive Analysis options here
    ]
    selected_graph = st.selectbox("Choose a graph to display:", ["Competitive Analysis Graphs"] + graph_options)
    if selected_graph:
        competitive_analysis(selected_graph)
elif menu == "ML Model Application":
    st.sidebar.title("ML Model Options")
    ml_option = st.sidebar.selectbox("Choose Model", ["Regression", "Classification"])
    if ml_option == "Regression":
        regression_model()
    elif ml_option == "Classification":
        classification_model()
