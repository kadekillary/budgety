#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List
from itertools import product
from datetime import datetime
import base64

from budgeotto import Budgeotto, CurveOpts, OptimizeMax

# TODO: only import what's absolutely necessary
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
import pandas as pd
import numpy as np

# from numpy import where, mean, median
from sklearn.metrics import r2_score

st.set_option("deprecation.showfileUploaderEncoding", False)

st.set_page_config(
    # TODO: change shark emoji
    page_title="Budget Optimization",
    page_icon=":shark:",
    layout="wide",
)

REDBOX_LOGO = "https://xmedia.com/images/logo_xm-redbox.png"

COLUMN_MAPPING = {
    "investment": "spend",
    "cost": "spend",
    "booked accounts": "target",
    "mta total completes": "target",
}


def get_time_from_columns(columns: List[str]) -> str:
    # Need to extract granularity in order to adjust optimization
    values = list(filter(lambda x: x in columns, ["day", "week", "month"]))
    # Filter will return the results in order. Therefore, we want to default
    # to whatever is the lowest level of granularity
    return values[0]


def remove_primary_columns(columns: List[str], time: str) -> List[str]:
    # Remove primary columns
    return [col for col in columns if col not in [time, "spend", "target", "month"]]


@st.cache(show_spinner=False, allow_output_mutation=True)
def preprocess(file_path: str):
    # TODO: make this cleaner
    # maybe re-write to chain
    # Skip first 6 rows -> due to Datorama format
    # data = pd.read_csv(file_path, skiprows=6, low_memory=False)
    data = pd.read_csv(file_path, low_memory=False)
    data.columns = data.columns.str.lower()
    data = data.rename(columns=COLUMN_MAPPING)
    time = get_time_from_columns(data.columns)
    data[time] = pd.to_datetime(data[time])
    # Convert -> `str` otherwise will get aggregated
    # Function -> convert to Q1, Q2 -> etc...
    data["month_selector"] = data[time].dt.month
    cols = remove_primary_columns(data.columns, time)
    return (data, cols, time)


# TODO: figure out cache / mutation
#  @st.cache(allow_output_mutation=True)
def get_dimension_curves(
    budgey: Budgeotto, minimum_rows: int, standard_deviation: int
) -> dict:
    return budgey.get_curves(min_rows=minimum_rows, std_dev=standard_deviation)


def apply_log_curve(
    df: pd.DataFrame, selected_dimension: str, column_name: str, curves: dict
) -> pd.DataFrame:
    print(curves)
    df[column_name] = df.apply(
        lambda row: CurveOpts.log_curve(
            row.spend,
            curves[row[selected_dimension]]["intercept"],
            curves[row[selected_dimension]]["coef"],
        ),
        axis=1,
    )
    return df


def create_pct_column(old_value: pd.Series, new_value: pd.Series):
    percent_change = ((new_value - old_value) / old_value) * 100
    return percent_change


def get_table_download_link(df: pd.DataFrame) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    # TODO: add time to results csv
    href = f'<a href="data:file/csv;base64,{b64}" download="optimization-results.csv">Download Results</a>'
    return href


# TODO: fix this shit - shouldn't re-run when selecting a new value
@st.cache(show_spinner=False)
def create_dimension_spend_range(
    dimensions: List[str], max_spend: int
) -> List[List[int]]:
    spend_range = (x for x in range(0, max_spend, int(max_spend / 100)))
    return list(product(dimensions, spend_range))


def add_filter(data, column, values):
    return data[data[column].isin(values)]


@st.cache(show_spinner=False)
def aggregate_data(data: pd.DataFrame, time: str, dimension: str) -> pd.DataFrame:
    return (
        data.groupby([time, dimension])[["spend", "target"]]
        .sum()
        .reset_index()
        .query("spend > 5")
    )


def convert_time(time: str, days: int) -> int:
    if time == "day":
        return days
    if time == "week":
        return days // 7
    if time == "month":
        return days // 30
    else:
        pass


def main():
    st.sidebar.image(REDBOX_LOGO)

    uploaded_file = st.sidebar.file_uploader("", type="csv")

    if uploaded_file is None:
        # TODO: add explanation info
        pass
    else:
        # To account for buffer reset
        # https://stackoverflow.com/questions/64347681/emptydataerror-no-columns-to-parse-from-file-about-streamlit
        uploaded_file.seek(0)
        data, columns, time = preprocess(uploaded_file)

        st.sidebar.subheader("Section")
        app_mode = st.sidebar.selectbox("", ["Explorer", "Predictor", "Optimizer"])
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)

        st.sidebar.subheader("Dimension")
        selected_dimension = st.sidebar.selectbox("", columns)

        st.sidebar.subheader("Month")
        q = list(data.month_selector.unique())
        month_selected = st.sidebar.multiselect("", q, default=q)
        data = data[data.month_selector.isin(month_selected)]

        st.sidebar.subheader("Filter")
        apply_filter = st.sidebar.checkbox("Apply Additional Filter")

        if apply_filter:
            column = st.sidebar.selectbox("Attribute", columns)
            values = st.sidebar.multiselect("Values", data[column].unique())
            if not values:
                st.warning("Please select a value to filter by!")
                st.stop()
            data = add_filter(data, column, values)

        # TODO: function -> cache it
        grouped_data = aggregate_data(data, time, selected_dimension)

        print(grouped_data)

        if grouped_data.shape[0] > 1:
            pass
        else:
            st.error("No Spend Error: there is no spend against this dimension!")
            st.stop()

        ##########################
        ######## EXPLORER ########
        ##########################
        if app_mode == "Explorer":
            sorted_data = grouped_data.sort_values(by=[time])

            spend = px.bar(
                sorted_data,
                x=time,
                y="spend",
                color=selected_dimension,
                template="plotly_white",
                labels={
                    time: "",
                    "spend": "",
                    selected_dimension: f"{selected_dimension.title()}",
                },
            ).update_yaxes(tickprefix="$")
            st.subheader(f"Spend - {time.title()}")
            st.subheader(f"Total: ${grouped_data.spend.sum():,.0f}")
            st.plotly_chart(spend, use_container_width=True)

            target = px.bar(
                sorted_data,
                x=time,
                y="target",
                color=selected_dimension,
                template="plotly_white",
                labels={
                    time: "",
                    "target": "",
                    selected_dimension: f"{selected_dimension.title()}",
                },
            )
            st.subheader(f"Target - {time.title()}")
            st.subheader(f"Total: {grouped_data.target.sum():,.0f}")
            st.plotly_chart(target, use_container_width=True)

        ##########################
        ####### PREDICTOR ########
        ##########################
        elif app_mode == "Predictor":
            sorted_data = grouped_data.sort_values(by=["spend"])

            budgeotto = Budgeotto(sorted_data, selected_dimension, time)

            # TODO: add try /except and callout for not enough values
            curves = get_dimension_curves(
                budgeotto, minimum_rows=5, standard_deviation=3
            )

            # Some dimensions may be removed due to having negative sloping
            # curves. Therefore, the only valid dimensions are the remaining
            # values.
            included_dimensions = list(curves.keys())

            spend_max = int(data.spend.max() * 0.8)
            curve_list = create_dimension_spend_range(included_dimensions, spend_max)

            # TODO: wrap into separate function
            df = pd.DataFrame(curve_list, columns=[selected_dimension, "spend"])
            _df = apply_log_curve(df, selected_dimension, "predicted_target", curves)
            _df = _df.query("predicted_target >= 0")

            all_curves = px.line(
                _df,
                x="spend",
                y="predicted_target",
                color=selected_dimension,
                template="plotly_white",
                labels={
                    "spend": "",
                    "predicted_target": "",
                    selected_dimension: f"{selected_dimension.title()}",
                },
            ).update_xaxes(tickprefix="$")

            st.plotly_chart(all_curves, use_container_width=True)

            st.markdown("&nbsp;", unsafe_allow_html=True)

            col1, col2 = st.beta_columns([1, 3])

            with col1:
                st.subheader("Value")
                selected_value = st.selectbox("", included_dimensions)
                filtered_data = sorted_data[
                    sorted_data[selected_dimension] == selected_value
                ]

                st.subheader(f"Spend - {time.title()}")
                spend = st.number_input(
                    "",
                    min_value=0,
                    max_value=10_000_000,
                    value=int(curves[selected_value]["median_spend"]),
                    step=1_000,
                )
                intercept = curves[selected_value]["intercept"]
                coef = curves[selected_value]["coef"]
                target = CurveOpts.log_curve(
                    spend,
                    intercept,
                    coef,
                )
                st.subheader(f"Prediction - {time.title()}")
                st.success(f"{target:,.0f}")

            print(f"Intercept: {curves[selected_value]['intercept']}")
            print(f"Coef: {curves[selected_value]['coef']}")

            with col2:
                filtered_data["predicted"] = filtered_data.apply(
                    lambda x: CurveOpts.log_curve(x["spend"], intercept, coef), axis=1
                )
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.spend,
                        y=filtered_data.target,
                        mode="markers",
                        name="Data",
                    )
                )
                fig.add_trace(
                    go.Line(
                        x=filtered_data.spend,
                        #  y=CurveOpts.log_curve(filtered_data.spend, intercept, coef),
                        y=filtered_data.predicted,
                        mode="lines",
                        name="Fit",
                    )
                )
                fig.update_layout(template="plotly_white").update_xaxes(tickprefix="$")

                st.plotly_chart(fig, use_container_width=True)

                st.subheader(
                    f"R2 Score: {r2_score(filtered_data.target, filtered_data.predicted):.2f}"
                )

        ##########################
        ####### OPTIMIZER ########
        ##########################
        elif app_mode == "Optimizer":
            budgeotto = Budgeotto(grouped_data, selected_dimension, time)
            # TODO: allow for optimizing across subset of dimension values
            # -> multiselect of values from dimension

            st.sidebar.subheader("Settings")

            curves = get_dimension_curves(
                budgeotto, minimum_rows=5, standard_deviation=3
            )
            # Some dimensions may be removed due to having negative sloping
            # curves. Therefore, the only valid dimensions are the remaining
            # values.
            included_dimensions = list(curves.keys())

            remove_dimensions = st.sidebar.multiselect(
                "Remove Dimension", included_dimensions
            )

            if remove_dimensions is not None:
                included_dimensions = [
                    dim for dim in included_dimensions if dim not in remove_dimensions
                ]

            remaining_data = grouped_data[
                grouped_data[selected_dimension].isin(included_dimensions)
            ]

            days = (remaining_data[time].max() - remaining_data[time].min()).days

            initial_time = convert_time(time, days)

            initial_budget = (
                int(
                    remaining_data.groupby([selected_dimension])["spend"].median().sum()
                )
                * initial_time
            )

            total_budget = st.sidebar.number_input("Total Budget", value=initial_budget)
            total_time = st.sidebar.number_input(
                f"Total {time.title()}s", value=initial_time
            )

            run_slot = st.sidebar.empty()

            aggregated_data = (
                grouped_data.groupby([selected_dimension])
                .agg({"spend": np.median, "target": np.mean})
                .reset_index()
            )
            aggregated_data["current_spend"] = aggregated_data.spend
            aggregated_data["current_target"] = aggregated_data.target

            dimension_spend_constr = budgeotto.create_dimension_attr()

            st.sidebar.subheader("Spend Variation (-/+)")
            st.sidebar.write(
                "The sliders represent the lower and upper bound relative to the total spend."
            )
            st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)

            for dimension in included_dimensions:
                current_total_spend = int(
                    curves[dimension]["median_spend"] * total_time
                )
                curves[dimension]["median_spend"] = (
                    int(
                        st.sidebar.number_input(
                            f"{dimension}", min_value=0, value=current_total_spend
                        )
                    )
                    / total_time
                )
                dimension_spend_constr[dimension] = st.sidebar.slider(
                    f"{dimension.title()} - Variation",
                    min_value=-100,
                    max_value=100,
                    step=1,
                    value=(-30, 30),
                    format="%g%%",
                )
                st.sidebar.markdown("&nbsp;", unsafe_allow_html=True)

            optimizer = OptimizeMax(
                curves,
                dimension_spend_constr,
                included_dimensions,
                total_budget // total_time,
            )

            run = run_slot.button("Run Optimization")
            if run:
                result = optimizer.run()

                # TODO: check that result was returned
                result.columns = [selected_dimension, "optimized_spend"]
                result["total_optimized_spend"] = result["optimized_spend"] * total_time
                result["predicted_target"] = result.apply(
                    lambda x: CurveOpts.log_curve(
                        x["optimized_spend"],
                        curves[x[selected_dimension]]["intercept"],
                        curves[x[selected_dimension]]["coef"],
                    ),
                    axis=1,
                )

                result = result.merge(
                    aggregated_data[
                        [selected_dimension, "current_spend", "current_target"]
                    ]
                )

                result = result.sort_values(by=[selected_dimension])

                spend = px.bar(
                    result,
                    x=selected_dimension,
                    y=["current_spend", "optimized_spend"],
                    barmode="group",
                    template="plotly_white",
                    labels={
                        selected_dimension: "",
                        "value": "",
                        "variable": f"Spend - {time.title()}",
                    },
                ).update_yaxes(tickprefix="$")
                st.plotly_chart(spend, use_container_width=True)

                target = px.bar(
                    result,
                    x=selected_dimension,
                    y=["current_target", "predicted_target"],
                    barmode="group",
                    template="plotly_white",
                    labels={
                        selected_dimension: "",
                        "value": "",
                        "variable": f"Target - {time.title()}",
                    },
                )
                st.plotly_chart(target, use_container_width=True)

                result["spend_pct"] = create_pct_column(
                    result["current_spend"], result["optimized_spend"]
                )
                result["target_pct"] = create_pct_column(
                    result["current_target"], result["predicted_target"]
                )
                result["target_pct"] = np.where(
                    np.isinf(result["target_pct"]),
                    result["predicted_target"].astype(int) * 100,
                    result["target_pct"],
                )

                result = result[
                    [
                        selected_dimension,
                        #  "current_spend",
                        "optimized_spend",
                        "spend_pct",
                        "total_optimized_spend",
                        "current_target",
                        "predicted_target",
                        "target_pct",
                    ]
                ]

                # TODO: usb specific for now
                result["cpa"] = np.where(
                    np.isinf(result["optimized_spend"] / result["predicted_target"]),
                    result["predicted_target"],
                    result["optimized_spend"] / result["predicted_target"],
                )

                result = result.rename(
                    columns={
                        selected_dimension: f"{selected_dimension.title()}",
                        "optimized_spend": "Recommended Spend",
                        "spend_pct": "Spend Change %",
                        "total_optimized_spend": "Total Recommended Spend",
                        "current_target": "Average Response",
                        "predicted_target": "Predicted Response",
                        "target_pct": "Response Change %",
                        "cpa": "CPA",
                    }
                )

                result = result.sort_values(by=["Recommended Spend"], ascending=False)

                result.iloc[:, 1:] = result.iloc[:, 1:].astype(int)

                st.markdown(get_table_download_link(result), unsafe_allow_html=True)

                fig = ff.create_table(result)
                st.plotly_chart(fig, use_container_width=True)


main()
