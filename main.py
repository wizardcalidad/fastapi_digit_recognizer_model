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
<style>
body {
  margin: 0;
  font-family: Arial, Helvetica, sans-serif;
}

.topnav {
  overflow: hidden;
  background-color: #333;
}

.topnav a {
  float: left;
  color: #f2f2f2;
  text-align: center;
  padding: 14px 16px;
  text-decoration: none;
  font-size: 17px;
}

.topnav a:hover {
  background-color: #ddd;
  color: black;
}

.topnav a.active {
  background-color: #04AA6D;
  color: white;
}
</style>
</head>
<body>

<div class="topnav">
  <a class="active" href="#home">Home</a>
  <a href="#about">About</a>
</div>

<div style="padding-left:16px">
  <h2>Digit Recognition app</h2>
  <p>Insert your image here</p>
  <form action="/predict_upload_file/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
</br></br>
<input type="submit">
</form>
</div>

</body>
</html>

    """

    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
