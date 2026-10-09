from flask import Flask, request, render_template
import joblib

obj = joblib.load("california.joblib")

model = obj["model"]
columns = obj["columns"]

app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        columns=columns
    )


@app.route("/predict", methods=["POST"])
def predict():

    input_data = []

    for column in columns:

        value = request.form.get(column)

        input_data.append(float(value))


    prediction = model.predict([input_data])

    # California dataset target is in $100,000 units
    result = prediction[0] * 100000


    return render_template(
        "index.html",
        columns=columns,
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)