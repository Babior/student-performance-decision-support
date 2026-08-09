"""
Streamlit application for the Decision-Support System.
"""

import streamlit as st


# Configure the browser tab and page layout.
st.set_page_config(
    page_title="Student Performance Support System",
    layout="wide",
)


# Create the navigation menu in the sidebar.
st.sidebar.title("Navigation")

selected_page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Student Prediction",
        "Model Comparison",
        "Data Visualisations",
        "About",
    ],
)


# Display the Home page.
if selected_page == "Home":
    st.title("Student Performance Decision-Support System")

    st.write(
        """
        This application helps teachers and school administrators identify
        students who may benefit from additional academic support.
        """
    )

    st.info(
        """
        This prediction is intended to support professional judgment.
        """
    )


# Display the Student Prediction page.
elif selected_page == "Student Prediction":
    st.title("Student Prediction")

    st.write(
        """
        Enter selected student information to estimate whether the student
        may require and benefit from additional academic support.
        """
    )

    st.warning("The prediction form will be added in the next stage.")


# Display the Model Comparison page.
elif selected_page == "Model Comparison":
    st.title("Model Comparison")

    st.write(
        """
        This page will compare the logistic regression, decision tree,
        and random forest models.
        """
    )

    st.warning("Model results have not yet been connected.")


# Display the Data Visualisations page.
elif selected_page == "Data Visualisations":
    st.title("Data Visualisations")

    st.write(
        """
        This page will display selected charts from the student-performance
        dataset.
        """
    )

    st.warning("Dataset charts will be added later.")


# Display the About page.
elif selected_page == "About":
    st.title("About This Project")

    st.write(
        """
        This system uses machine learning to support the early identification
        of students who may need additional academic assistance.
        """
    )

    st.subheader("Important limitations")

    st.write(
        """
        The dataset comes from two Portuguese secondary schools.
        Results may not generalise directly to Ghanaian schools,
        universities, or other educational settings.
        """
    )