document.addEventListener('DOMContentLoaded', function () {
    // Event listener for "Generate Outfit" button
    document.getElementById('generateOutfit').addEventListener('click', function() {
        // Changes the source of the displayed image to Image 1
        document.getElementById('displayedImage').src = 'https://via.placeholder.com/300?text=Image+1';
    });

    // Event listener for form submission
    document.getElementById('weatherForm').addEventListener('submit', function(event) {
        event.preventDefault(); // Prevents the default form submission action
        console.log('Form submitted.'); // Check console for "Form submitted" message

        // Get selected weather condition and number
        const weatherSelect = document.getElementById('weather');
        const weatherCondition = weatherSelect.value;
        const numberInput = document.getElementById('number');
        const numberValue = parseInt(numberInput.value, 10);
        const timeInput = document.getElementById('time');
        const timeValue = timeInput.value;
        const timeParts = timeValue.split(":");
        const hours = parseInt(timeParts[0]);
        console.log('Weather condition:', weatherCondition);

        const greetingSpan = document.getElementById('greeting');
        const messageSpan = document.getElementById('message');

        if (hours >= 5 && hours < 12) {
            greetingSpan.textContent = "Good Morning!";
            messageSpan.textContent = "Let's set you up with an outfit for the day";
        } else if (hours >= 12 && hours < 18) {
            greetingSpan.textContent = "Good Afternoon!";
            messageSpan.textContent = "Let's set you up with an outfit for the day";
        } else if (hours >= 18 && hours < 21) {
            greetingSpan.textContent = "Good Evening!";
            messageSpan.textContent = "Let's set you up with an outfit for a night out";
        } else {
            greetingSpan.textContent = "Good Night!";
            messageSpan.textContent = "Let's set you up with an outfit for a night out";
        }

        // Save weather condition to localStorage for other pages to access
        localStorage.setItem('weatherCondition', weatherCondition);

        // Change background based on weather condition and number
        const body = document.body;
        if (weatherCondition === 'sunny') {
            body.style.backgroundImage = 'none'; // Reset any existing background image
            body.style.backgroundColor = '#aad7f6'; // Set background color if needed
            body.style.backgroundImage = 'url("sunny.jpg")'; // Set sunny background image
            body.style.backgroundSize = 'cover';
        }
        else if (weatherCondition === 'cloudy') {
            body.style.backgroundImage = 'none'; // Reset any existing background image
            body.style.backgroundColor = '#aad7f6'; // Set background color if needed
            body.style.backgroundImage = 'url("cloudy.jpg")'; // Set cloudy background image
            body.style.backgroundSize = 'cover';
        }
        else if (weatherCondition === 'rainy') {
            body.style.backgroundImage = 'none'; // Reset any existing background image
            body.style.backgroundColor = '#aad7f6'; // Set background color if needed
            body.style.backgroundImage = 'url("rainy.jpg")'; // Set rainy background image
            body.style.backgroundSize = 'cover';
        }

        // Reset form
        weatherSelect.selectedIndex = 0;
        numberInput.value = '';
        timeInput.value = '';
    });
});

function setBackgroundImage() {
    // Get the weather condition from localStorage (set by index.html)
    const weatherCondition = localStorage.getItem('weatherCondition');
    console.log('Weather condition:', localStorage.getItem('weatherCondition'));

    // Set background image based on weather condition
    const body = document.body;
    if (weatherCondition === 'sunny') {
        body.style.backgroundImage = 'url("sunny.jpg")';
    } else if (weatherCondition === 'cloudy') {
        body.style.backgroundImage = 'url("cloudy.jpg")';
    } else if (weatherCondition === 'rainy') {
        body.style.backgroundImage = 'url("rainy.jpg")';
    }
    body.style.backgroundSize = 'cover';
}
