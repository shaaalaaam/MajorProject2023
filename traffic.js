document.addEventListener('DOMContentLoaded', function () {
    console.log("My JavaScript file is running.");
    const counts = document.querySelectorAll('.count');
    const images = document.querySelectorAll('.image');
    const imageLane0=document.querySelector('image-lane0');

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
    function updateImageSource() {
        // Update the source with the current index from the array
        var img = document.getElementById('image-lane0');
        // img.src = 'DjangoProject/trafficLight/traffic_light_app/lane_0.jpg';
        imageLane0.src = 'DjangoProject/trafficLight/traffic_light_app/lane_0.jpg';
        console.log("Update Image source")
    }
    setInterval(() => {
        // Fetch the count from the API endpoint for lane1
        fetch('http://127.0.0.1:9000/get-count0')
            .then(response => response.text())
            .then(count => {
                // Update the count of lane1
                counts[0].textContent = count;
            })
            .catch(error => console.log(error));
            
        fetch('http://127.0.0.1:9000/get-count1')
            .then(response => response.text())
            .then(count => {
                // Update the count of lane1
                counts[1].textContent = count;
            })
            .catch(error => console.log(error));
            
        fetch('http://127.0.0.1:9000/get-count2')
            .then(response => response.text())
            .then(count => {
                // Update the count of lane1
                counts[2].textContent = count;
            })
            .catch(error => console.log(error));
            
        fetch('http://127.0.0.1:9000/get-count3')
            .then(response => response.text())
            .then(count => {
                // Update the count of lane1
                counts[3].textContent = count;
            })
            .catch(error => console.log(error));
            console.log(counts[0]);
            console.log(counts[1]);
            console.log(counts[2]);
            console.log(counts[3]);
            console.log('\n');
            
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
        updateImageSource();
    }, 1000);
});
