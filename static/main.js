function run() {
    var uploadForm = document.getElementById('uploadForm');
    var formData = new FormData(uploadForm);

    $.ajax({
        url: "/inferences",
        type: "POST",
        data: formData,
        processData: false,
        contentType: false,
        success: function (resp) {
            var textArea = document.getElementById('resultText');
            textArea.innerHTML = "";
            for (i in resp["results"]) {
                var res = resp["results"][i];
                var bbox = res[0];
                var text = res[1];
                var score = Number((res[2] * 100).toFixed(2));
                textArea.innerHTML += text + "\n";
            }
            var content = "data:image/jpeg;base64, " + resp.img;
            document.getElementById("resultImg").src = content;
        },
    });
}
