from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/tables")
def tables_report():
    return render_template("tables_report.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)