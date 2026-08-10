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


elif selected_page == "Student Prediction":
    st.title("Student Prediction")

    st.write(
        """
        Enter student information below to estimate whether the student
        may benefit from additional academic support.
        """
    )

    st.info(
        """
        The prediction form is being prepared for integration with the
        final trained machine-learning pipeline.
        """
    )

    st.subheader("Student Information")

    with st.form("student_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            school = st.selectbox(
                "School",
                ["Select an option", "GP", "MS"],
            )

            sex = st.selectbox(
                "Sex",
                ["Select an option", "Female", "Male"],
            )

            age = st.number_input(
                "Age",
                min_value=15,
                max_value=22,
                value=17,
                step=1,
            )

        with col2:
            study_time = st.selectbox(
                "Weekly Study Time",
                [
                    "Select an option",
                    "Less than 2 hours",
                    "2 to 5 hours",
                    "5 to 10 hours",
                    "More than 10 hours",
                ],
            )

            failures = st.number_input(
                "Number of Previous Class Failures",
                min_value=0,
                max_value=4,
                value=0,
                step=1,
            )

            absences = st.number_input(
                "Number of Absences",
                min_value=0,
                value=0,
                step=1,
            )

        submitted = st.form_submit_button("Generate Prediction")

    if submitted:
        if (
        school == "Select an option"
        or sex == "Select an option"
        or study_time == "Select an option"
    ):
            st.error(
            "Please complete all required fields before generating a prediction."
        )
        else:
            st.success("Student information submitted successfully.")

        st.warning(
            """
            Prediction is not yet available because the final trained model
            has not been connected to the application.
            """
        )

    st.divider()

    st.caption(
        """
        This prediction is intended to support professional judgment.
        It should not be used as the sole basis for academic,
        disciplinary, or counselling decisions.
        """
    )


elif selected_page == "Model Comparison":
    st.title("Model Comparison")

    st.write(
        """
        Compare the performance of the machine-learning models developed
        for the student performance decision-support system.
        """
    )

    st.info(
        """
        Final evaluation results will be displayed here once the trained
        models and evaluation outputs are integrated into the application.
        """
    )

    st.subheader("Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Logistic Regression")
        st.write(
            """
            A baseline classification model that estimates the probability
            that a student may benefit from additional academic support.
            """
        )

    with col2:
        st.markdown("### Decision Tree")
        st.write(
            """
            A rule-based model that makes predictions by splitting students
            into groups based on their characteristics.
            """
        )

    with col3:
        st.markdown("### Random Forest")
        st.write(
            """
            An ensemble model that combines multiple decision trees to
            produce a more stable prediction.
            """
        )

    st.divider()

    st.subheader("Evaluation Metrics")

    st.write(
        """
        The final models will be compared using the following metrics:
        """
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Support Recall", "Pending")
        st.metric("Support F1-Score", "Pending")

    with metric_col2:
        st.metric("Testing Accuracy", "Pending")
        st.metric("ROC-AUC", "Pending")

    with metric_col3:
        st.metric("Cross-Validation Mean", "Pending")
        st.metric("Cross-Validation Std.", "Pending")

    st.caption(
        """
        Special attention will be given to recall for students who may
        require academic support because missing such a student could delay
        necessary intervention.
        """
    )

    st.divider()

    st.subheader("Detailed Results")

    st.warning(
        """
        Model comparison tables, confusion matrices, and final evaluation
        results will appear here after integration with the team's model
        outputs.
        """
    )


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