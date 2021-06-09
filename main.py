import uvicorn
from PIL import Image
from io import BytesIO
from starlette.responses import HTMLResponse
from fastapi import FastAPI, File, UploadFile
from util_digits import convert_image_to_array
from tensorflow.keras.models import load_model


app = FastAPI(title="Digit image recognition model")


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.get("/ping")
def ping():
    return "pong"


@app.post("/predict_upload_file/")
async def digit_image(file: UploadFile = File(...)):
    model = load_model("digit_reconizer.h5")
    image_array = convert_image_to_array(file.file, 28, 28, 1)
    predictions = int(model.predict_classes(image_array))
    return predictions


@app.get("/")
async def index():
    content = """
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
</head>
<body>

<nav class="navbar navbar-inverse">
  <div class="container-fluid">
    <div class="navbar-header">
      <a class="navbar-brand" href="#">Digit Recognizer</a>
    </div>
    <ul class="nav navbar-nav">
      <li class="active"><a href="#">Home</a></li>
      <li><a href="#">Services</a></li>
      <li><a href="#">About Us</a></li>
    </ul>
    <button class="btn btn-danger navbar-btn">Button</button>
  </div>
</nav>

<div style="padding-left:16px">
  <h2>Digit Recognition app</h2>
  <p>Insert your image here</p>
  <form action="/predict_upload_file/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
</br></br>
<input type="submit">
</form>
</div>
<footer>
<div>this project is done by Qoyum Yusuf @github.com/wizardcalidad.</div>
</footer>
</body>
</html>

    """

    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
