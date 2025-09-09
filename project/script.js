const fileInput = document.getElementById('butterflyImage');
const fileName = document.getElementById('fileName');

fileInput?.addEventListener('change', function () {
    fileName.textContent = this.files.length > 0 ? this.files[0].name : 'No file chosen';
});
