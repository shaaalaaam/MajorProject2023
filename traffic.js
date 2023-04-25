document.addEventListener('DOMContentLoaded', function () {
    console.log("My JavaScript file is running.");
    const counts = document.querySelectorAll('.count');
    const images = document.querySelectorAll('.image');

    let greenIndex = 0;
    let thresholdCount = 15;
    let previousGreenIndex = 0;
    let blinkGreen = false;
    let blinkDuration = 1000; // Blinking duration in milliseconds

    function updateImages() {
        // Convert the NodeList to an array for sorting
        const countsArray = Array.from(counts);

        // Sort the counts array in descending order
        countsArray.sort((a, b) => parseInt(b.textContent) - parseInt(a.textContent));

        // Check if the lane with the most vehicles should have green light
        if (parseInt(countsArray[0].textContent) > thresholdCount) {
            greenIndex = countsArray.findIndex(count => parseInt(count.textContent) === parseInt(countsArray[0].textContent));
        } else {
            greenIndex = (previousGreenIndex + 1) % countsArray.length;
        }

        // Loop through the counts array and set the src property of the images accordingly
        countsArray.forEach((count, index) => {
            if (index === greenIndex) {
                if (parseInt(counts[greenIndex].textContent) === 0) {
                    // Green lane count is zero, blink the light
                    if (blinkGreen) {
                        images[index].src = 'images/yellow.png';
                    } else {
                        images[index].src = 'images/green.png';
                    }
                    blinkGreen = !blinkGreen; // toggle blinkGreen flag
                } else {
                    images[index].src = 'images/green.png';
                }
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
        counts.forEach((count, index) => {
            let currentCount = parseInt(count.textContent);
            if (index === greenIndex && currentCount > 0) {
                count.textContent = currentCount - 1;
            } else {
                count.textContent = Math.floor(Math.random() * 20);
            }
        });
        if (parseInt(counts[greenIndex].textContent) === 0) {
            // Blink the green light for blinkDuration milliseconds
            setInterval(() => {
                if (blinkGreen) {
                    images[greenIndex].src = 'images/yellow.png';
                } else {
                    images[greenIndex].src = 'images/green.png';
                }
                blinkGreen = !blinkGreen; // toggle blinkGreen flag
            }, blinkDuration);
            greenIndex = (greenIndex + 1) % counts.length;
        }
        updateImages();
    }, 2000);

    updateImages();
});
