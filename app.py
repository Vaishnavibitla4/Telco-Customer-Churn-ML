import gradio as gr
from src.serving.inference import predict  # import your production-ready predict

# UI
interface = gr.Interface(
    fn=lambda *args: predict({
        "gender": args[0],
        "SeniorCitizen": 1 if args[1] == "Yes" else 0,
        "Partner": args[2],
        "Dependents": args[3],
        "PhoneService": args[4],
        "MultipleLines": args[5],
        "InternetService": args[6],
        "OnlineSecurity": args[7],
        "OnlineBackup": args[8],
        "DeviceProtection": args[9],
        "TechSupport": args[10],
        "StreamingTV": args[11],
        "StreamingMovies": args[12],
        "Contract": args[13],
        "PaperlessBilling": args[14],
        "PaymentMethod": args[15],
        "tenure": args[16],
        "MonthlyCharges": args[17],
        "TotalCharges": args[18],
    }),
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Yes", "No"], label="Senior Citizen"), 
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["No", "Yes", "No phone service"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown([
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ], label="Payment Method"),
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges")
    ],
    outputs="text",
    title="📊 Telco Customer Churn Prediction",
    description="Fill in customer details to predict churn probability"
)

if __name__ == "__main__":
    interface.launch()