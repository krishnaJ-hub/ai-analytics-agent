import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(page_title="AI Analytics Agent",page_icon="🤖")
st.config.set_option('server.maxUploadSize', 1024)
st.title("🤖 AI Analytics Agent")
st.write("Welcome! upload your data file let AI do the analysis for you!")
st.markdown("---")
st.subheader("📂 Upload your data file")
uploaded_file=st.file_uploader("Choose a csv or excel file",type=["csv","xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df=pd.read_csv(uploaded_file, encoding='latin-1')
    else:
        df=pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.subheader("📊 Your data")
    st.dataframe(df) 
    st.markdown("---")
    st.subheader("🔍 Data profile Report")
    col1,col2,col3,col4=st.columns(4)
    with col1:
        st.metric("Total Rows",df.shape[0])
    with col2:
        st.metric("Total Columns",df.shape[1])
    with col3:
        st.metric("Missing Values",df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows",df.duplicated().sum())
    st.markdown("---")
    st.subheader("📋 Column Information")
    st.dataframe(df.dtypes.reset_index().rename(columns={0: "Data Type","index": "Column Name"}),hide_index=True)
    st.markdown("---")
    st.subheader("🧹 Automated Data Cleaning")
    original_df=df.copy()
    df=df.drop_duplicates()
    missing_percent=(df.isnull().sum().sum()/df.size)*100
    st.write("Missing percentage:",round(missing_percent,2))
    if missing_percent<5:
        df=df.dropna()
    else:
        for col in df.select_dtypes(include='number').columns:
            df[col]=df[col].fillna(df[col].median())
        for col in df.select_dtypes(include='object').columns:
            df[col]=df[col].fillna("Unknown")
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col]=pd.to_datetime(df[col])
            except:
                pass
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col]=pd.to_datetime(df[col], 
                      infer_datetime_format=True,
                      dayfirst=True)          
        except:
            pass            
    for col in df.select_dtypes(include='object').columns:
        df[col]=df[col].str.strip()
    rows_after=len(df)
    rows_before=len(original_df)
    duplicates_removed=rows_before-rows_after
    st.success("✅ Data Cleaning Completed!")
    col1,col2 =st.columns(2)
    with col1:
        st.metric("Rows Before",rows_before)
    with col2:
        st.metric("Rows After",rows_after)  
    st.write("✅ Duplicates Removed:",duplicates_removed) 
    st.write("✅ Data types fixed for date columns")
    st.write("✅ Whitespace cleaned from text columns")
    st.write("✅ Missing values handled using 5% rule")
    st.markdown("---")
    st.subheader("🤖 AI Powered Insights")
    from groq import Groq
    import os
    from dotenv import load_dotenv
    load_dotenv()  
    client=Groq(api_key=os.getenv("GROQ_API_KEY"))
    summary=f"""
    Dataset has {df.shape[0]} rows and {df.shape[1]} columns.
    Columnnames: {list(df.columns)}
    Statistics:
    {df.describe().to_string()}
    """
    prompt=f"""
    You are an elite data analyst and strategic 
business consultant with 20+ years of 
experience working with Fortune 500 companies.

You have been given a dataset to analyse.
Your job is to provide world class analysis
that helps business leaders make smart decisions.

DATASET INFORMATION:
{summary}

Please provide the following analysis:

🔍 EXECUTIVE SUMMARY
Write 2-3 sentences summarising 
what this dataset is about and 
its overall health!

📊 TOP 5 DATA INSIGHTS
For each insight:
- What you found
- Why it matters to the business
- The exact numbers that prove it

📈 TOP 3 GROWTH OPPORTUNITIES  
For each opportunity:
- What the opportunity is
- How to act on it
- Expected business impact

⚠️ TOP 2 RISK WARNINGS
For each risk:
- What the risk is
- Why it is dangerous
- What to do immediately

💡 ONE GOLDEN RECOMMENDATION
The single most important action
this business should take right now!

Rules:
- Be specific with numbers from the data
- Think like a CEO reading this report
- Every insight must be actionable
- No vague statements — only facts!

but use good and understandable english and make it more explanable way 
"""
    with st.spinner("🤖 AI is Analysing your data please wait..."):
        response=client.chat.completions.create(
             model="groq/compound-mini",
             messages=[{"role":"user","content": prompt}])                           
            
        insights=response.choices[0].message.content
    st.markdown("### 🤖 AI Analysis report") 
    st.write(insights)
    st.markdown("---")
    st.subheader("📊 Auto Visulisations")
    st.markdown("#### 📊 Category Distribution Charts") 
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique()<15:
            fig=px.bar(
                df[col].value_counts().reset_index(),
                x=col,
                y="count",
                title=f"Distribution of {col}",
                color=col
            )
            st.plotly_chart(fig)
    st.markdown("#### Numerical Distribution Charts")
    skip_cols = ['row_id', 'id', 'postal_code',
                 'zip', 'index']
    for col in df.select_dtypes(include="number").columns:
        if any(skip in col.lower()
               for skip in skip_cols):
            continue
        fig=px.histogram(
            df,
            x=col,
            title=f"Distribution of {col}",
            color_discrete_sequence=['#636EFA']
        )
        st.plotly_chart(fig)
    st.markdown("#### 📅 Trend Charts")
    date_cols=df.select_dtypes(include='datetime').columns.tolist()
    num_cols=df.select_dtypes(include='number').columns.tolist()
    if len(date_cols)>0 and len(num_cols)>0:
        skip_cols = ['row_id', 'id', 'postal_code',
                     'zip', 'index']
        for date_col in date_cols:
            for num_col in num_cols:
                if any(skip in num_col.lower()
                   for skip in skip_cols):
                    continue
                trend_data = df.groupby(
                    df[date_col].dt.to_period('M')
            )[num_col].sum().reset_index()
            trend_data[date_col] = trend_data[date_col].astype(str)
            fig=px.line(
                trend_data,
                x=date_col,
                y=num_col,
                title=f"{num_col} by {date_cols}",
                markers=True
            )
            st.plotly_chart(fig)
    else:
        st.info("ℹ️ No date columns for Trend Analysis")
    st.markdown("#### 🔥 Correlation Heatmap")
    skip_cols = ['row_id', 'id', 'postal_code',
                 'zip', 'index','postal code']
    num_df = df.select_dtypes(include='number')
    num_df = num_df[[
        col for col in num_df.columns
        if num_df[col].nunique() / len(num_df) < 0.5
        and not any(skip in col.lower()
                    for skip in skip_cols)
    ]]
    corr_matrix = num_df.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        title="Correlation Between Numerical Columns"
    )
    st.plotly_chart(fig)
    st.markdown("---")
    st.subheader("📥 Download Your Results")
    csv=df.to_csv(index=False)
    st.download_button(
        label="📥 Download Clean data (CSV)",
        data=csv,
        file_name="Cleaned_data.csv",
        mime="text/csv"
    ) 
    report=f"""
    AI ANALYTICS REPORT
    {'='*50}
    
    DATA SUMMARY:
    TOTAL ROWS={df.shape[0]}
    TOTAL COLUMNS={df.shape[1]}
    
    AI INSIGHTS:
    {insights}
    
    Report generated by AI ANALYTICS AGENT """
    st.download_button(
        label="📄 Download Insights Report",
        data=report,
        file_name="insights_report.txt",
        mime="text/plain"
    )
    
    
       
         
                     
else:
    st.info("👆 Please upload your data file not to waste your time analysing on your own")   
           
