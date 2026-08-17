"""
Streamlit application for the Student Performance Decision-Support System.
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.utils import (
    ADDRESS_OPTIONS,
    EDUCATION_OPTIONS,
    FAMILY_SIZE_OPTIONS,
    FIVE_POINT_OPTIONS,
    GUARDIAN_OPTIONS,
    HEALTH_OPTIONS,
    JOB_OPTIONS,
    PARENT_STATUS_OPTIONS,
    REASON_OPTIONS,
    RELATIONSHIP_OPTIONS,
    SEX_OPTIONS,
    STUDY_TIME_OPTIONS,
    TRAVEL_TIME_OPTIONS,
    YES_NO_OPTIONS,
    build_student_dataframe,
    load_final_model,
    load_model_metadata,
)


# Page configuration
st.set_page_config(
    page_title="Student Performance Support System",
    layout="wide",
)


# Model loading
@st.cache_resource
def get_final_model():
    """Load and cache the final machine-learning pipeline."""
    return load_final_model()


@st.cache_data
def get_final_metadata():
    """Load and cache the final model metadata."""
    return load_model_metadata()


# Sidebar navigation
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


# Home page
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


# Student Prediction page
elif selected_page == "Student Prediction":
    st.title("Student Prediction")

    st.write(
        """
        Enter student information below to estimate whether the student
        may benefit from additional academic support.
        """
    )

    # Load final model and metadata
    try:
        final_model = get_final_model()
        model_metadata = get_final_metadata()

    except Exception as error:
        st.error(
            "The prediction model could not be loaded. "
            "Please check the model files and try again."
        )
        st.exception(error)
        st.stop()

    st.info(
        """
        Complete the sections below and select Generate Prediction
        to receive the model's decision-support result.
        """
    )

    st.subheader("Student Information")

    # Student prediction form
    with st.form("student_prediction_form"):

        # Section 1: Academic Progress
        with st.expander(
            "1. Academic Progress",
            expanded=True,
        ):
            st.caption(
                "Enter two academic results available before the student's "
                "final course grade."
            )

            col1, col2 = st.columns(2)

            with col1:
                g1 = st.number_input(
                    "Earlier Academic Result",
                    min_value=0,
                    max_value=19,
                    value=10,
                    step=1,
                    help=(
                        "Enter an earlier assessment or grading result "
                        "from the current course."
                    ),
                )

                study_time_label = st.selectbox(
                    "Weekly study time",
                    list(STUDY_TIME_OPTIONS.keys()),
                )

                failures = st.number_input(
                    "Previous class failures",
                    min_value=0,
                    max_value=3,
                    value=0,
                    step=1,
                )

            with col2:
                g2 = st.number_input(
                    "Most Recent Academic Result",
                    min_value=0,
                    max_value=19,
                    value=10,
                    step=1,
                    help=(
                        "Enter the student's latest available assessment "
                        "result before the final course grade."
                    ),
                )

                absences = st.number_input(
                    "Number of absences",
                    min_value=0,
                    max_value=32,
                    value=0,
                    step=1,
                )

        # Section 2: Student & School
        with st.expander(
            "2. Student & School",
        ):
            col1, col2 = st.columns(2)

            with col1:
                school = st.selectbox(
                    "School",
                    ["GP", "MS"],
                )

                sex_label = st.selectbox(
                    "Sex",
                    list(SEX_OPTIONS.keys()),
                )

                age = st.number_input(
                    "Age",
                    min_value=15,
                    max_value=22,
                    value=17,
                    step=1,
                )

            with col2:
                address_label = st.selectbox(
                    "Home area",
                    list(ADDRESS_OPTIONS.keys()),
                )

                travel_time_label = st.selectbox(
                    "Travel time to school",
                    list(TRAVEL_TIME_OPTIONS.keys()),
                )

                reason_label = st.selectbox(
                    "Main reason for choosing the school",
                    list(REASON_OPTIONS.keys()),
                )

        # Section 3: Family Background
        with st.expander(
            "3. Family Background",
        ):
            col1, col2 = st.columns(2)

            with col1:
                mother_education_label = st.selectbox(
                    "Mother's education level",
                    list(EDUCATION_OPTIONS.keys()),
                )

                mother_job_label = st.selectbox(
                    "Mother's occupation",
                    list(JOB_OPTIONS.keys()),
                )

                family_size_label = st.selectbox(
                    "Family size",
                    list(FAMILY_SIZE_OPTIONS.keys()),
                )

                guardian_label = st.selectbox(
                    "Primary guardian",
                    list(GUARDIAN_OPTIONS.keys()),
                )

            with col2:
                father_education_label = st.selectbox(
                    "Father's education level",
                    list(EDUCATION_OPTIONS.keys()),
                )

                father_job_label = st.selectbox(
                    "Father's occupation",
                    list(JOB_OPTIONS.keys()),
                )

                parent_status_label = st.selectbox(
                    "Parents' living arrangement",
                    list(PARENT_STATUS_OPTIONS.keys()),
                )

                family_relationship_label = st.selectbox(
                    "Quality of family relationships",
                    list(RELATIONSHIP_OPTIONS.keys()),
                )

        # Section 4: Support & Resources
        with st.expander(
            "4. Support & Resources",
        ):
            col1, col2 = st.columns(2)

            with col1:
                school_support_label = st.selectbox(
                    "Extra educational support from school",
                    list(YES_NO_OPTIONS.keys()),
                )

                family_support_label = st.selectbox(
                    "Educational support from family",
                    list(YES_NO_OPTIONS.keys()),
                )

                paid_classes_label = st.selectbox(
                    "Extra paid classes",
                    list(YES_NO_OPTIONS.keys()),
                )

                activities_label = st.selectbox(
                    "Participates in extracurricular activities",
                    list(YES_NO_OPTIONS.keys()),
                )

            with col2:
                nursery_label = st.selectbox(
                    "Attended nursery school",
                    list(YES_NO_OPTIONS.keys()),
                )

                higher_label = st.selectbox(
                    "Plans to pursue higher education",
                    list(YES_NO_OPTIONS.keys()),
                )

                internet_label = st.selectbox(
                    "Internet access at home",
                    list(YES_NO_OPTIONS.keys()),
                )

        # Section 5: Lifestyle & Wellbeing
        with st.expander(
            "5. Lifestyle & Wellbeing",
        ):
            col1, col2 = st.columns(2)

            with col1:
                free_time_label = st.selectbox(
                    "Free time after school",
                    list(FIVE_POINT_OPTIONS.keys()),
                )

                going_out_label = st.selectbox(
                    "Frequency of going out with friends",
                    list(FIVE_POINT_OPTIONS.keys()),
                )

                weekday_alcohol_label = st.selectbox(
                    "Weekday alcohol consumption",
                    list(FIVE_POINT_OPTIONS.keys()),
                )

            with col2:
                weekend_alcohol_label = st.selectbox(
                    "Weekend alcohol consumption",
                    list(FIVE_POINT_OPTIONS.keys()),
                )

                health_label = st.selectbox(
                    "Current health status",
                    list(HEALTH_OPTIONS.keys()),
                )

                romantic_label = st.selectbox(
                    "Currently in a romantic relationship",
                    list(YES_NO_OPTIONS.keys()),
                )

        submitted = st.form_submit_button(
            "Generate Prediction"
        )

    # Generate prediction
    if submitted:
        student_data = {
            "school": school,
            "sex": SEX_OPTIONS[sex_label],
            "age": int(age),
            "address": ADDRESS_OPTIONS[address_label],
            "famsize": FAMILY_SIZE_OPTIONS[family_size_label],
            "Pstatus": PARENT_STATUS_OPTIONS[parent_status_label],
            "Medu": EDUCATION_OPTIONS[mother_education_label],
            "Fedu": EDUCATION_OPTIONS[father_education_label],
            "Mjob": JOB_OPTIONS[mother_job_label],
            "Fjob": JOB_OPTIONS[father_job_label],
            "reason": REASON_OPTIONS[reason_label],
            "guardian": GUARDIAN_OPTIONS[guardian_label],
            "traveltime": TRAVEL_TIME_OPTIONS[travel_time_label],
            "studytime": STUDY_TIME_OPTIONS[study_time_label],
            "failures": int(failures),
            "schoolsup": YES_NO_OPTIONS[school_support_label],
            "famsup": YES_NO_OPTIONS[family_support_label],
            "paid": YES_NO_OPTIONS[paid_classes_label],
            "activities": YES_NO_OPTIONS[activities_label],
            "nursery": YES_NO_OPTIONS[nursery_label],
            "higher": YES_NO_OPTIONS[higher_label],
            "internet": YES_NO_OPTIONS[internet_label],
            "romantic": YES_NO_OPTIONS[romantic_label],
            "famrel": RELATIONSHIP_OPTIONS[family_relationship_label],
            "freetime": FIVE_POINT_OPTIONS[free_time_label],
            "goout": FIVE_POINT_OPTIONS[going_out_label],
            "Dalc": FIVE_POINT_OPTIONS[weekday_alcohol_label],
            "Walc": FIVE_POINT_OPTIONS[weekend_alcohol_label],
            "health": HEALTH_OPTIONS[health_label],
            "absences": int(absences),
            "G1": int(g1),
            "G2": int(g2),
        }

        student_df = build_student_dataframe(student_data)

        expected_features = (
            model_metadata["numeric_features"]
            + model_metadata["categorical_features"]
        )

        missing_features = (
            set(expected_features)
            - set(student_df.columns)
        )

        extra_features = (
            set(student_df.columns)
            - set(expected_features)
        )

        if missing_features or extra_features:
            st.error(
                "The student information does not match the features "
                "expected by the trained model."
            )

        else:
            try:
                support_class = model_metadata["positive_class"]

                class_index = list(
                    final_model.classes_
                ).index(support_class)

                support_probability = float(
                    final_model.predict_proba(
                        student_df
                    )[0][class_index]
                )

                threshold = float(
                    model_metadata["threshold"]
                )

                st.subheader("Prediction Result")

                if support_probability >= threshold:
                    st.warning(
                        "May benefit from additional academic support"
                    )

                else:
                    st.success(
                        "Likely to pass"
                    )

                st.metric(
                    "Estimated probability of needing support",
                    f"{support_probability:.1%}",
                )

            except Exception as error:
                st.error(
                    "A prediction could not be generated. "
                    "Please check the student information and try again."
                )
                st.exception(error)

    st.divider()

    st.caption(
        """
        This prediction is intended to support professional judgment.
        It should not be used as the sole basis for academic,
        disciplinary, or counselling decisions.
        """
    )


# Model Comparison page
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
        Model selection prioritised the ability to identify students who
        may need academic support, while also considering F1-score,
        ROC-AUC, cross-validation stability, overfitting and
        interpretability.
        """
    )

    st.subheader("Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Logistic Regression")

        st.write(
            """
            A classification model that estimates the probability that a
            student may benefit from additional academic support.
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

    # Final model results
    try:
        model_metadata = get_final_metadata()

        st.subheader("Selected Final Model")

        st.write(
            """
            The selected deployment model is the progress-informed
            Logistic Regression model.
            """
        )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric(
                "Support Recall",
                f"{model_metadata['test_recall']:.1%}",
            )

            st.metric(
                "Support F1-Score",
                f"{model_metadata['test_f1']:.1%}",
            )

        with metric_col2:
            st.metric(
                "Testing Accuracy",
                f"{model_metadata['test_accuracy']:.1%}",
            )

            st.metric(
                "ROC-AUC",
                f"{model_metadata['test_roc_auc']:.3f}",
            )

        with metric_col3:
            st.metric(
                "Decision Threshold",
                f"{model_metadata['threshold']:.2f}",
            )

            st.metric(
                "Input Features",
                len(
                    model_metadata["numeric_features"]
                    + model_metadata["categorical_features"]
                ),
            )

        st.caption(
            """
            Special attention is given to recall for students who may require
            academic support because missing such a student could delay
            necessary intervention.
            """
        )

        st.divider()

        # Confusion matrix
        st.subheader("Confusion Matrix")

        confusion_matrix = model_metadata["confusion_matrix"]

        confusion_df = pd.DataFrame(
            confusion_matrix,
            index=[
                "Actual: Likely to pass",
                "Actual: Needs support",
            ],
            columns=[
                "Predicted: Likely to pass",
                "Predicted: Needs support",
            ],
        )

        st.table(confusion_df)

        st.caption(
            """
            The confusion matrix shows how the final model performed on the
            held-out test data. Correct predictions appear on the diagonal.
            """
        )

    except Exception as error:
        st.error(
            "The final model evaluation results could not be displayed."
        )
        st.exception(error)


# Data Visualisations page
elif selected_page == "Data Visualisations":
    st.title("Data Visualisations")

    st.write(
        """
        Explore selected patterns in the student-performance dataset used
        to develop the decision-support system.
        """
    )

    try:
        student_data = pd.read_csv(
            "data/raw/student-por.csv",
            sep=";",
        )

        student_data["needs_support"] = (
            student_data["G3"] < 10
        ).astype(int)

        student_data["Support Status"] = (
            student_data["needs_support"]
            .map(
                {
                    0: "Likely to pass",
                    1: "May require academic support",
                }
            )
        )

        # Student outcome distribution
        st.subheader("Student Outcome Distribution")

        outcome_counts = (
            student_data["Support Status"]
            .value_counts()
            .reset_index()
        )

        outcome_counts.columns = [
            "Support Status",
            "Number of Students",
        ]

        outcome_chart = (
            alt.Chart(outcome_counts)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Support Status:N",
                    title="Student Outcome",
                    axis=alt.Axis(
                        labelAngle=0,
                        labelLimit=300,
                    ),
                ),
                y=alt.Y(
                    "Number of Students:Q",
                    title="Number of Students",
                ),
                tooltip=[
                    "Support Status",
                    "Number of Students",
                ],
            )
        )

        st.altair_chart(
            outcome_chart,
            width="stretch",
        )

        st.caption(
            """
            This chart shows the number of students in each outcome group.
            The support group contains students whose final grade was below
            the project's passing threshold.
            """
        )

        st.divider()

        # Academic results comparison
        st.subheader("Academic Results by Support Status")

        academic_summary = (
            student_data
            .groupby("Support Status")[["G1", "G2"]]
            .mean()
            .rename(
                columns={
                    "G1": "Earlier Academic Result",
                    "G2": "Most Recent Academic Result",
                }
            )
            .reset_index()
        )

        academic_long = academic_summary.melt(
            id_vars="Support Status",
            var_name="Academic Measure",
            value_name="Average Result",
        )

        academic_chart = (
            alt.Chart(academic_long)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Support Status:N",
                    title="Student Outcome",
                    axis=alt.Axis(
                        labelAngle=0,
                        labelLimit=300,
                    ),
                ),
                y=alt.Y(
                    "Average Result:Q",
                    title="Average Academic Result",
                ),
                xOffset="Academic Measure:N",
                color=alt.Color(
                    "Academic Measure:N",
                    title="Academic Measure",
                ),
                tooltip=[
                    "Support Status",
                    "Academic Measure",
                    alt.Tooltip(
                        "Average Result:Q",
                        format=".2f",
                    ),
                ],
            )
        )

        st.altair_chart(
            academic_chart,
            width="stretch",
        )

        st.caption(
            """
            This chart compares the average earlier and most recent
            academic results for students in each outcome group.
            """
        )

        st.divider()

        # Study time comparison
        st.subheader("Study Time by Support Status")

        study_time_labels = {
            1: "Less than 2 hours",
            2: "2 to 5 hours",
            3: "5 to 10 hours",
            4: "More than 10 hours",
        }

        study_time_summary = (
            student_data
            .groupby(
                [
                    "studytime",
                    "Support Status",
                ]
            )
            .size()
            .reset_index(
                name="Number of Students"
            )
        )

        study_time_summary["Study Time"] = (
            study_time_summary["studytime"]
            .map(study_time_labels)
        )

        study_time_chart = (
            alt.Chart(study_time_summary)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Study Time:N",
                    title="Weekly Study Time",
                    sort=[
                        "Less than 2 hours",
                        "2 to 5 hours",
                        "5 to 10 hours",
                        "More than 10 hours",
                    ],
                    axis=alt.Axis(
                        labelAngle=0,
                        labelLimit=200,
                    ),
                ),
                y=alt.Y(
                    "Number of Students:Q",
                    title="Number of Students",
                ),
                xOffset="Support Status:N",
                color=alt.Color(
                    "Support Status:N",
                    title="Student Outcome",
                ),
                tooltip=[
                    "Study Time",
                    "Support Status",
                    "Number of Students",
                ],
            )
        )

        st.altair_chart(
            study_time_chart,
            width="stretch",
        )

        st.caption(
            """
            This chart compares weekly study-time categories for students
            who were likely to pass and students who may require academic
            support.
            """
        )

    except Exception as error:
        st.error(
            "The dataset could not be loaded for visualisation."
        )
        st.exception(error)


# About page
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