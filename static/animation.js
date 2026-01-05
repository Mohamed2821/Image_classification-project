function predictImage() {
    const input = document.getElementById("imageInput");
    const loader = document.getElementById("loader");
    const results = document.getElementById("results");

    if (!input.files.length) {
        alert("Please upload an image");
        return;
    }

    loader.style.display = "block";
    results.innerHTML = "";

    const formData = new FormData();
    formData.append("image", input.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loader.style.display = "none";

        if (data.error) {
            results.innerHTML = `<p style="color:red">${data.error}</p>`;
            return;
        }

        data.predictions.forEach(p => {
            results.innerHTML += `
                <div class="result">
                    <strong>${p.label}</strong> — ${p.confidence.toFixed(2)}%
                </div>
            `;
        });
    })
    .catch(() => {
        loader.style.display = "none";
        results.innerHTML = "<p style='color:red'>Prediction failed</p>";
    });
}
