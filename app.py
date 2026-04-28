import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="PromoPulse Dashboard", layout="wide")

st.title("PromoPulse: NLP-Driven Sales Nowcasting")
st.write("A lightweight dashboard for forecasting results and NLP impact analysis.")

# Load data
summary_overall = pd.read_csv("summary_overall.csv")
summary_by_horizon = pd.read_csv("summary_by_horizon.csv")
compare_pivot = pd.read_csv("compare_pivot.csv")
forecast_detail = pd.read_csv("forecast_detailed_predictions_with_nlp.csv")
forecast_detail["Date"] = pd.to_datetime(forecast_detail["Date"])
forecast_detail["origin"] = pd.to_datetime(forecast_detail["origin"])

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

model_options = ["All"] + sorted(summary_overall["model"].unique().tolist())
selected_model = st.sidebar.selectbox("Choose model", model_options)

# -----------------------------
# Apply Filters
# -----------------------------
summary_overall_filtered = summary_overall.copy()
summary_by_horizon_filtered = summary_by_horizon.copy()
compare_pivot_filtered = compare_pivot.copy()

if selected_model != "All":
    summary_overall_filtered = summary_overall_filtered[
        summary_overall_filtered["model"] == selected_model
    ]
    summary_by_horizon_filtered = summary_by_horizon_filtered[
        summary_by_horizon_filtered["model"] == selected_model
    ]
    compare_pivot_filtered = compare_pivot_filtered[
        compare_pivot_filtered["model"] == selected_model
    ]

# -----------------------------
# Quick Summary
# -----------------------------
st.subheader("Quick Summary")

if not summary_overall_filtered.empty:
    best_row = summary_overall_filtered.loc[summary_overall_filtered["sMAPE"].idxmin()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", f"{best_row['model']}")
    col2.metric("Feature Setting", f"{best_row['feature_set']}")
    col3.metric("Lowest sMAPE", f"{best_row['sMAPE']:.4f}")
else:
    st.info("No data available for the selected filters.")

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(
    ["Forecast Summary", "History vs NLP Analysis", "NLP Signal Explorer"]
)

# -----------------------------
# Tab 1: Forecast Summary
# -----------------------------
with tab1:
    st.subheader("Overall Results")
    st.dataframe(summary_overall_filtered, use_container_width=True)

    st.subheader("Results by Horizon")
    st.dataframe(summary_by_horizon_filtered, use_container_width=True)

    st.subheader("sMAPE Illustration")

    if selected_model == "All":
        st.info("Please select one model from the sidebar to view the chart clearly.")
    elif not summary_by_horizon_filtered.empty:
        chart_df = summary_by_horizon_filtered[["horizon", "feature_set", "sMAPE"]].copy()
        chart_df["label"] = chart_df["feature_set"] + " (h=" + chart_df["horizon"].astype(str) + ")"

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(chart_df["label"], chart_df["sMAPE"])

        y_min = chart_df["sMAPE"].min()
        y_max = chart_df["sMAPE"].max()
        margin = max((y_max - y_min) * 0.2, 0.1)

        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_ylabel("sMAPE")
        ax.set_xlabel("Feature Setting and Horizon")
        ax.set_title(f"sMAPE Comparison for {selected_model}")

        for i, v in enumerate(chart_df["sMAPE"]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

        plt.xticks(rotation=20)
        st.pyplot(fig)

    else:
        st.info("No chart available for the selected filters.")

# -----------------------------
# Tab 2: History vs NLP
# -----------------------------
with tab2:
    st.subheader("HistoryOnly vs History+NLP Comparison")
    st.dataframe(compare_pivot_filtered, use_container_width=True)

    if not compare_pivot_filtered.empty:
        row = compare_pivot_filtered.iloc[0]

        if "delta_sMAPE" in compare_pivot_filtered.columns:
            delta = row["delta_sMAPE"]

            if delta < 0:
                st.success(
                    f"For {row['model']}, History+NLP achieved lower sMAPE than HistoryOnly by {abs(delta):.4f}."
                )
            elif delta > 0:
                st.warning(
                    f"For {row['model']}, History+NLP was worse than HistoryOnly by {delta:.4f} sMAPE."
                )
            else:
                st.info(
                    f"For {row['model']}, History+NLP and HistoryOnly performed the same on sMAPE."
                )
    else:
        st.info("No comparison insight available.")

    st.markdown("---")
    st.subheader("Forecast Demo (RandomForest)")

    st.write(
        "This demo compares Actual Sales, HistoryOnly forecast, and History+NLP forecast "
        "for selected horizon using RandomForest."
    )

    demo_model = "RandomForest"

    demo_horizon_options = sorted(
        forecast_detail.loc[
            forecast_detail["model"] == demo_model, "horizon"
        ].astype(int).unique().tolist()
    )
    demo_horizon = st.selectbox("Select horizon", demo_horizon_options, key="demo_horizon")

    compare_demo_df = forecast_detail[
        (forecast_detail["model"] == demo_model) &
        (forecast_detail["horizon"] == demo_horizon) &
        (forecast_detail["feature_set"].isin(["HistoryOnly", "History+NLP"]))
    ].copy()

    if compare_demo_df.empty:
        st.info("No forecast demo data available for this selection.")
    else:
        preferred_stores = [150, 200, 250, 300, 350, 400]
        store_options = [s for s in preferred_stores if s in compare_demo_df["Store"].unique().tolist()]

        if not store_options:
            st.info("None of the preferred stores are available for this horizon.")
        else:
            demo_store = st.selectbox("Select store", store_options, key="demo_store")

            compare_demo_df = compare_demo_df[
                compare_demo_df["Store"] == demo_store
            ].copy()

            compare_demo_df = compare_demo_df.sort_values("origin")

            pred_pivot = compare_demo_df.pivot_table(
                index="origin",
                columns="feature_set",
                values="predicted_sales",
                aggfunc="mean"
            ).reset_index()

            actual_df = compare_demo_df.groupby("origin", as_index=False)["actual_sales"].mean()
            plot_df = actual_df.merge(pred_pivot, on="origin", how="left")

            st.write("### Comparison Data")
            st.dataframe(plot_df, use_container_width=True)

            st.write("### Actual vs HistoryOnly vs History+NLP")

            def calc_smape(y_true, y_pred):
                denom = (abs(y_true) + abs(y_pred)) / 2
                smape = abs(y_true - y_pred) / denom
                smape = smape.replace([float("inf")], 0).fillna(0)
                return smape.mean() * 100

            col1, col2 = st.columns(2)

            if "HistoryOnly" in plot_df.columns:
                smape_hist = calc_smape(plot_df["actual_sales"], plot_df["HistoryOnly"])
                col1.metric("Demo sMAPE: HistoryOnly", f"{smape_hist:.2f}")

            if "History+NLP" in plot_df.columns:
                smape_nlp = calc_smape(plot_df["actual_sales"], plot_df["History+NLP"])
                col2.metric("Demo sMAPE: History+NLP", f"{smape_nlp:.2f}")
            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                plot_df["origin"],
                plot_df["actual_sales"],
                marker="o",
                label="Actual Sales"
            )

            if "HistoryOnly" in plot_df.columns:
                ax.plot(
                    plot_df["origin"],
                    plot_df["HistoryOnly"],
                    marker="o",
                    label="HistoryOnly Prediction"
                )

            if "History+NLP" in plot_df.columns:
                ax.plot(
                    plot_df["origin"],
                    plot_df["History+NLP"],
                    marker="o",
                    label="History+NLP Prediction"
                )

            ax.set_title(f"Store {demo_store} | RandomForest | h={demo_horizon}")
            ax.set_xlabel("Forecast Origin Date")
            ax.set_ylabel("Sales")
            ax.legend()
            plt.xticks(rotation=30)
            st.pyplot(fig)
                
# -----------------------------
# Tab 3: NLP Signal Explorer
# -----------------------------
with tab3:
    st.subheader("NLP Signal Explorer")
    user_review = st.text_area("Enter a sample review")

    promo_keywords = {"discount", "voucher", "promo", "free delivery", "sale"}
    complaint_keywords = {"late", "lambat", "refund", "cancel", "slow", "problem", "bad"}

    positive_words = {"good", "nice", "great", "excellent", "best", "fast", "love"}
    negative_words = {"bad", "slow", "late", "refund", "cancel", "problem", "worst", "hate", "lambat"}

    def simple_clean(text):
        return text.lower().strip()

    def keyword_flag(text, keyword_set):
        return any(k in text for k in keyword_set)

    def simple_sentiment(text):
        pos_count = sum(word in text for word in positive_words)
        neg_count = sum(word in text for word in negative_words)

        if pos_count > neg_count:
            return "Positive"
        elif neg_count > pos_count:
            return "Negative"
        else:
            return "Neutral / Unclear"

    if st.button("Analyse Review"):
        cleaned = simple_clean(user_review)

        promo_flag = keyword_flag(cleaned, promo_keywords)
        complaint_flag = keyword_flag(cleaned, complaint_keywords)
        sentiment_label = simple_sentiment(cleaned)

        st.write("**Cleaned text:**", cleaned)
        st.write("**Sentiment:**", sentiment_label)
        st.write("**Promo-intent matched:**", "Yes" if promo_flag else "No")
        st.write("**Complaint cue matched:**", "Yes" if complaint_flag else "No")
