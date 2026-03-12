from flask import Flask, request, jsonify, Response
from parser.extract import parse_report

API_KEY = "ithrive_secure_2026_key"

app = Flask(__name__)


@app.route("/", methods=["GET"])
def health():
    return {
        "engine": "ithrive_disease_parser_v1",
        "status": "ok"
    }


@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok"}


@app.route("/parse-report", methods=["POST"])
def parse():

    auth = request.headers.get("Authorization")

    if auth != f"Bearer {API_KEY}":
        return jsonify({"error": "unauthorized"}), 401

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    pdf_file = request.files["file"]
    pdf_bytes = pdf_file.read()

    debug = request.args.get("debug")

    try:

        if debug == "1":
            png = parse_report(pdf_bytes, debug=True)
            return Response(
                png,
                mimetype="image/png"
            )

        result = parse_report(pdf_bytes)

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": "parser_failure",
            "message": str(e)
        }), 500


@app.route("/debug-crop", methods=["POST"])
def debug_crop():

    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    pdf_file = request.files["file"]
    pdf_bytes = pdf_file.read()

    png = parse_report(pdf_bytes, debug=True)

    return Response(
        png,
        mimetype="image/png"
    )


@app.route("/docs", methods=["GET"])
def docs():

    return """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>ITHRIVE Parser Test</title>

<style>
body{
    font-family: Arial;
    background:#111;
    color:#fff;
    padding:40px;
}

h1{
    color:#42BAB2;
}

button{
    padding:10px 20px;
    background:#42BAB2;
    border:none;
    color:white;
    font-size:16px;
    cursor:pointer;
}

input{
    margin:20px 0;
}

pre{
    background:#222;
    padding:20px;
    border-radius:6px;
    overflow:auto;
}
</style>

</head>

<body>

<h1>ITHRIVE Disease Parser</h1>

<input type="file" id="file">
<br>

<button onclick="upload()">Parse Report</button>

<h2>Response</h2>
<pre id="output">Waiting for upload...</pre>

<script>

async function upload(){

    const file = document.getElementById("file").files[0]

    if(!file){
        alert("Select a PDF")
        return
    }

    const formData = new FormData()
    formData.append("file", file)

    const res = await fetch("/parse-report",{
        method:"POST",
        headers:{
            "Authorization":"Bearer ithrive_secure_2026_key"
        },
        body:formData
    })

    const data = await res.json()

    document.getElementById("output").textContent =
        JSON.stringify(data,null,2)
}

</script>

</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
