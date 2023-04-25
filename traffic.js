document.addEventListener('DOMContentLoaded', function () {
    console.log("My JavaScript file is running.");
    const counts = document.querySelectorAll('.count');
    const images = document.querySelectorAll('.image');

    let greenIndex = 0;
    let thresholdCount=15;
    function updateImages() {
        // Convert the NodeList to an array for sorting
        const countsArray = Array.from(counts);

        // Sort the counts array in descending order
        countsArray.sort((a, b) => parseInt(b.textContent) - parseInt(a.textContent));

        // Check if the lane with the most vehicles should have green light
        let greenIndex = 0;
        if (parseInt(countsArray[0].textContent) > thresholdCount ) {
            greenIndex = countsArray.findIndex(count => parseInt(count.textContent) === parseInt(countsArray[0].textContent));
        } else {
            greenIndex = (previousGreenIndex + 1) % countsArray.length;
        }

        // Loop through the counts array and set the src property of the images accordingly
        countsArray.forEach((count, index) => {
            if (index === greenIndex) {
                images[index].src = 'images/green.png';
            } else if (index === (greenIndex + 1) % countsArray.length) {
                images[index].src = 'images/yellow.png';
            } else {
                images[index].src = 'images/red.png';
            }
        });

        // Update the previousGreenIndex for the next call to updateImages()
        previousGreenIndex = greenIndex;
    }




    setInterval(() => {
        counts.forEach((count) => {
            count.textContent = Math.floor(Math.random() * 20);
        });
        updateImages();
    }, 2000);

    updateImages();
});
