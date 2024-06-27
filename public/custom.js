document.addEventListener('DOMContentLoaded', function() {
  function addCustomWatermark() {
    const parentElement = document.querySelector('.MuiBox-root.css-mww0i9');

    if (parentElement) {
      console.log('Parent element found:', parentElement);

      // Ensure the original watermark is hidden
      const oldWatermark = parentElement.querySelector('.MuiStack-root.watermark.css-1705j0v');
      if (oldWatermark) {
        oldWatermark.style.display = 'none';
        console.log('Old watermark hidden');
      }

      // Create and insert the new watermark
      const newDiv = document.createElement('div');
      newDiv.className = 'custom-watermark';

      const newText = document.createElement('span');
      newText.textContent = "The assistant can make mistakes and provide inaccurate information. It's a simple demo.";

      newDiv.appendChild(newText);
      parentElement.appendChild(newDiv);

      console.log('New watermark added:', newDiv);
    } else {
      console.log('Parent element not found. Retrying...');
      setTimeout(addCustomWatermark, 500); // Retry after 500ms
    }
  }

  addCustomWatermark();
});