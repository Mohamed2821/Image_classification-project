function uploadImage() {
    let fileInput = document.getElementById("imageInput");
    let loader = document.getElementById("loader");
    let resultDiv = document.getElementById("result");

    if (!fileInput.files.length) {
        alert("Please upload an image");
        return;
    }

    let formData = new FormData();
    formData.append("image", fileInput.files[0]);

    loader.style.display = "block";
    resultDiv.innerHTML = "";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loader.style.display = "none";
        data.forEach(item => {
            resultDiv.innerHTML += `<p>${item.label} — ${item.confidence}</p>`;
        });
    })
    .catch(err => {
        loader.style.display = "none";
        resultDiv.innerHTML = "Error occurred";
    });
}
