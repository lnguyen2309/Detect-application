let currentSelectedDiv = null;

function selectImage(imageName, divElement) {
  // Store selected image name in hidden input
  document.getElementById('selectedImage').value = imageName;

  // Remove highlight from previously selected
  if (currentSelectedDiv) {
    currentSelectedDiv.classList.remove('selected');
  }

  // Highlight current selected
  divElement.classList.add('selected');
  currentSelectedDiv = divElement;
}

function runDetection(imageName = null) {
  const conf = document.getElementById('chiffre1').value;
  const iou = document.getElementById('chiffre2').value;
  const selectedImage = imageName || document.getElementById('selectedImage').value;
  const resultContent = document.getElementById('resultContent');
  const detectBtn = document.querySelector('.detect-btn');

  if (!selectedImage) {
    alert('Please select or upload an image first!');
    return;
  }

  detectBtn.disabled = true;
  detectBtn.textContent = 'En cours...';
  resultContent.innerHTML = '<p class="loading">En cours...</p>';

  fetch('/detect', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conf, iou, image: selectedImage })
  })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        const outputImageUrl = `/output/predict/${encodeURIComponent(data.output_image)}`;
        resultContent.innerHTML = `
          <div class="detected-image-container">
            <img src="${outputImageUrl}" alt="Detection Result" style="max-width: 100%; height: auto; border: 1px solid #ccc; margin-bottom: 15px;" />
          </div>
        `;
      } else {
        resultContent.innerHTML = `<div class="error">Error: ${data.error}</div>`;
      }
    })
    .catch(err => {
      resultContent.innerHTML = `<div class="error">Error: ${err.message}</div>`;
    })
    .finally(() => {
      detectBtn.disabled = false;
      detectBtn.textContent = 'Détecter';
    });
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    const previewImage = document.getElementById('previewImage');
    previewImage.src = e.target.result;
    previewImage.style.display = 'block';
  };
  reader.readAsDataURL(file);

  // Upload to backend
  uploadImageToServer(file);
}

function uploadImageToServer(file) {
  const formData = new FormData();
  formData.append('image', file);

  fetch('/upload', {
    method: 'POST',
    body: formData,
  })
    .then(res => res.json())
    .then(data => {
      if (data.success) {
        const uploadedFileName = data.filename;
        document.getElementById('selectedImage').value = uploadedFileName;

        // Clear any previous selection from gallery
        if (currentSelectedDiv) {
          currentSelectedDiv.classList.remove('selected');
          currentSelectedDiv = null;
        }

        // Show preview (optional)
        const previewImg = document.getElementById('previewImage');
        previewImg.src = URL.createObjectURL(file);
        previewImg.style.display = 'block';

        // ❌ Do NOT trigger detection here anymore!
      } else {
        alert('Upload failed: ' + data.error);
      }
    })
    .catch(err => {
      console.error('Upload error:', err);
      alert('Error uploading file');
    });
}

