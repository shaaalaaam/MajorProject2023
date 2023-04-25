document.addEventListener('DOMContentLoaded', function () {
    console.log("My JavaScript file is running.");
    const counts = document.querySelectorAll('.count');
    const images = document.querySelectorAll('.image');

    function updateImages() {
        counts.forEach((count, index) => {
            const value = parseInt(count.textContent);
            if (value === 0) {
                images[index].src = 'images/red.png';
            } else if (value < 5) {
                images[index].src = 'images/yellow.png';
            } else {
                images[index].src = 'images/green.png';
            }
        });
    }

    setInterval(() => {
        counts.forEach((count) => {
            count.textContent = Math.floor(Math.random() * 20);
        });
        updateImages();
    }, 2000);

    updateImages();
});
