# Student Performance Decision-Support System

## Overview

The Student Performance Decision-Support System is a machine-learning project developed for Introduction to Artificial Intelligence COurse.

The system is designed to help teachers and school administrators identify students who may benefit from additional academic support.

It is a decision-support tool. Its predictions are intended to support professional judgment, not replace it.

## Project Objectives

The project aims to:

- prepare and explore a student-performance dataset;
- create a binary target for identifying students who may need support;
- compare Logistic Regression, Decision Tree, and Random Forest models;
- compare early-warning and progress-informed prediction approaches;
- evaluate models using appropriate classification metrics;
- examine model performance and overfitting;
- deploy the selected model in a Streamlit application;
- present predictions, probabilities, model results, and selected visualisations responsibly.

## Dataset

The project uses the Portuguese-language portion of the UCI Student Performance Dataset.

The dataset contains:

- 649 student records;
- 33 original variables;
- academic information;
- attendance information;
- study habits;
- family background;
- school support;
- lifestyle information;
- access to educational resources.

The raw dataset is stored at:

`data/raw/student-por.csv`

## Target Variable

The project creates a binary target called:

`needs_support`

The target is defined as:

```python
needs_support = 1 if G3 < 10 else 0