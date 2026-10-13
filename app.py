import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Interactive VGG Research Dashboard", page_icon="🧠", layout="wide")
st.title("🧠 Interactive VGG Research Dashboard")
st.caption("CIFAR-10 | VGG-11, VGG-13, VGG-16 | Batch Normalization comparison")
st.info("This dashboard visualizes recorded results. Filters update the analysis; they do not retrain models or create new accuracy values.")

results = pd.DataFrame([
    {"Model":"VGG11","BatchNorm":"No","Train Accuracy (%)":78.18,"Validation Accuracy (%)":78.00,"Test Accuracy (%)":77.71,"Parameters":9225610},
    {"Model":"VGG11","BatchNorm":"Yes","Train Accuracy (%)":67.42,"Validation Accuracy (%)":68.94,"Test Accuracy (%)":68.96,"Parameters":9231114},
    {"Model":"VGG13","BatchNorm":"No","Train Accuracy (%)":74.12,"Validation Accuracy (%)":75.24,"Test Accuracy (%)":75.11,"Parameters":9410122},
    {"Model":"VGG13","BatchNorm":"Yes","Train Accuracy (%)":37.99,"Validation Accuracy (%)":38.94,"Test Accuracy (%)":39.79,"Parameters":9416010},
    {"Model":"VGG16","BatchNorm":"No","Train Accuracy (%)":71.36,"Validation Accuracy (%)":70.84,"Test Accuracy (%)":70.85,"Parameters":14719818},
    {"Model":"VGG16","BatchNorm":"Yes","Train Accuracy (%)":32.38,"Validation Accuracy (%)":32.48,"Test Accuracy (%)":33.32,"Parameters":14728266},
])

st.sidebar.header("Explore experiments")
results_file = st.sidebar.file_uploader("Optional: replace results with a CSV", type="csv")
history_file = st.sidebar.file_uploader("Optional: upload training-history CSV", type="csv")

if results_file:
    try:
        uploaded = pd.read_csv(results_file)
        # Map common result column spellings.
        aliases = {
            "Model":["model","model_name","experiment","experiment_name"],
            "BatchNorm":["batchnorm","batch_norm","bn"],
            "Train Accuracy (%)":["train_accuracy","train_acc","train accuracy"],
            "Validation Accuracy (%)":["validation_accuracy","val_accuracy","val_acc"],
            "Test Accuracy (%)":["test_accuracy","test_acc"],
            "Parameters":["parameters","params","num_parameters","parameter_count"],
        }
        lookup = {str(c).strip().lower():c for c in uploaded.columns}
        ren = {}
        for target, names in aliases.items():
            for name in names:
                if name in lookup and target not in uploaded.columns:
                    ren[lookup[name]] = target
                    break
        uploaded = uploaded.rename(columns=ren)
        if "Model" in uploaded.columns and any(c in uploaded.columns for c in ["Test Accuracy (%)","Validation Accuracy (%)"]):
            if "BatchNorm" not in uploaded.columns: uploaded["BatchNorm"] = "Unknown"
            for c in ["Train Accuracy (%)","Validation Accuracy (%)","Test Accuracy (%)","Parameters"]:
                if c not in uploaded.columns: uploaded[c] = pd.NA
                uploaded[c] = pd.to_numeric(uploaded[c], errors="coerce")
                if "Accuracy" in c and uploaded[c].dropna().size and uploaded[c].dropna().max() <= 1:
                    uploaded[c] *= 100
            results = uploaded
        else:
            st.sidebar.warning("Could not identify Model and accuracy columns; using built-in results.")
    except Exception as err:
        st.sidebar.error(f"Could not read CSV: {err}")

results["Model"] = results["Model"].astype(str)
results["BatchNorm"] = results["BatchNorm"].fillna("Unknown").astype(str)
models = sorted(results["Model"].unique())
bn_values = sorted(results["BatchNorm"].unique())
selected_models = st.sidebar.multiselect("VGG architecture", models, default=models)
selected_bn = st.sidebar.multiselect("BatchNorm", bn_values, default=bn_values)
data = results[results["Model"].isin(selected_models) & results["BatchNorm"].isin(selected_bn)].copy()

if data.empty:
    st.warning("No results match those filters. Select at least one model and BatchNorm option.")
    st.stop()

metric = "Test Accuracy (%)" if data["Test Accuracy (%)"].notna().any() else "Validation Accuracy (%)"
best = data.loc[data[metric].idxmax()] if data[metric].notna().any() else None
c1,c2,c3,c4 = st.columns(4)
c1.metric("Configurations", len(data))
c2.metric("Best accuracy", f"{best[metric]:.2f}%" if best is not None else "N/A")
c3.metric("Best configuration", f"{best['Model']} / BN {best['BatchNorm']}" if best is not None else "N/A")
params = pd.to_numeric(data["Parameters"], errors="coerce").dropna()
c4.metric("Largest model", f"{int(params.max()):,} parameters" if not params.empty else "N/A")

st.subheader("1. Results table")
st.dataframe(data.sort_values(metric, ascending=False, na_position="last"), use_container_width=True, hide_index=True)

st.subheader("2. Compare accuracy")
accuracy_cols = [c for c in ["Train Accuracy (%)","Validation Accuracy (%)","Test Accuracy (%)"] if c in data and data[c].notna().any()]
if accuracy_cols:
    long = data.melt(id_vars=["Model","BatchNorm"], value_vars=accuracy_cols, var_name="Metric", value_name="Accuracy (%)").dropna()
    long["Configuration"] = long["Model"] + " | BN " + long["BatchNorm"]
    fig = px.bar(long, x="Configuration", y="Accuracy (%)", color="Metric", barmode="group", text_auto=".1f")
    fig.update_layout(yaxis_range=[0,100], xaxis_title="", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

if data["Parameters"].notna().any() and data[metric].notna().any():
    st.subheader("3. Model size vs accuracy")
    scatter = data.dropna(subset=["Parameters",metric]).copy()
    scatter["Configuration"] = scatter["Model"] + " | BN " + scatter["BatchNorm"]
    fig2 = px.scatter(scatter, x="Parameters", y=metric, color="Model", symbol="BatchNorm", hover_name="Configuration")
    fig2.update_layout(yaxis_range=[0,100], xaxis_title="Number of parameters")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("4. Training curves (optional)")
if history_file:
    try:
        hist = pd.read_csv(history_file)
        low = {str(c).lower():c for c in hist.columns}
        epoch = next((low[k] for k in ["epoch","epochs"] if k in low), None)
        if epoch is None:
            hist.insert(0, "Epoch", range(1,len(hist)+1)); epoch = "Epoch"
        metrics = [c for c in hist.columns if any(x in str(c).lower() for x in ["loss","accuracy","acc"])]
        if metrics:
            chosen = st.selectbox("Training-history metric", metrics)
            st.plotly_chart(px.line(hist, x=epoch, y=chosen, markers=True), use_container_width=True)
        else:
            st.warning("No loss/accuracy columns detected in this CSV.")
        with st.expander("Show uploaded training history"): st.dataframe(hist, use_container_width=True)
    except Exception as err:
        st.error(f"Could not read history CSV: {err}")
else:
    st.caption("Upload final_training_history.csv in the sidebar to view epoch-by-epoch curves.")

st.subheader("5. Research notes")
st.markdown("""
- The built-in values are from the recorded final five-epoch experiments.
- These results describe this particular run and do **not** prove BatchNorm is generally worse.
- For stronger conclusions, train longer, tune settings consistently, and repeat experiments with multiple random seeds.
""")
st.download_button("Download filtered results CSV", data=data.to_csv(index=False).encode("utf-8"), file_name="filtered_vgg_results.csv", mime="text/csv")
with st.expander("Dashboard limitations"):
    st.write("This app visualizes results; it does not retrain the neural networks, load .pth weights, or classify new images.")
