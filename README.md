# OutfitOracle

OutfitOracle is an innovative project that combines the power of Python and Node.js to provide a 
comprehensive solution for the need to spend few minutes everyday looking in your closet, not knowing what to wear,
while also keeping it stylish, and fit to the weather.

Given the user’s wardrobe, and the weather for the day, the system
suggests an outfit. With time, the system will learn about the user’s style and
preferences and will be able to suggest better outfits.

## Prerequisites

Before you get started, ensure you have the following installed on your system:

- [Git](https://git-scm.com/downloads)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- [Node.js](https://nodejs.org/en/download/)

## Installation

To set up OutfitOracle on your local machine, follow these steps:

### 1. Clone the Repository

Clone the OutfitOracle repository to your local machine and navigate into it:

```bash
git clone https://github.com/TkuiTku100/OutfitOracle.git
cd path/to/project
```

### 2. Set Up the Conda Environment

Create and activate the Conda environment from the outfit_oracle.yml file to install all required Python dependencies:

```bash
conda env create -f outfit_oracle.yml
conda activate outfitoracle
```

### 3. Run the Python Server

Within the activated Conda environment, navigate to the project folder and start the Python server:

```bash
python pred.py
```

### 4. Start the Node.js Server

Open a new terminal window, activate the Conda environment again, and run the Node.js server:

```bash
cd path/to/project
conda activate outfitoracle
node server.js
```

### 5. Open the Welcome Page

Lastly, in a new terminal window, activate the Conda environment, navigate to the .idea folder within the project, and open the welcome.html file in your default browser:

```bash
cd path/to/project/
cd .idea
conda activate outfitoracle
start welcome.html  # Use 'open welcome.html' on macOS.
```

# User Instructions for OutfitOracle

Once you've successfully set up and started the OutfitOracle project by following the installation instructions, here's how to use the application:

## Logging In

OutfitOracle provides two predefined user profiles you can use to log in and explore different functionalities:

1. **Username**: `alon` - A male profile with a complete wardrobe and training data.
2. **Username**: `inbar` - A female profile with a complete wardrobe and basic training data.
3. **Username**: `your username` - You can add yourself to the system as well, but don't forget to add your wardrobe as well.

Choose one of these usernames to log in and access the home page.

## After Logging In

Once you're on the home page, follow these steps to interact with the application:

### Selecting the Weather

1. Look for the weather selection widget in the bottom right corner of the home page.
2. Select the current weather conditions. This information helps the system generate appropriate outfit suggestions.

### Generating an Outfit

- Click on the **Generate Outfit** button to get your first outfit suggestion based on the selected weather and your profile's wardrobe and preferences.

### Not Pleased with the Suggestion?

If the suggested outfit doesn't meet your expectations, you have several options:

- **Regenerate**: Click on **Regenerate** to get a new outfit suggestion.
- **Change Top**: Click on **Change Top** if you're only dissatisfied with the top piece of the outfit.
- **Change Bottom**: Click on **Change Bottom** to replace only the bottom piece of the outfit.
- **Change Overtop**: Use **Change Overtop** to swap out only the overtop garment.

### Exploring the Wardrobe

You can view items in the wardrobe by:

1. Clicking on the **Wardrobe** button to navigate to the wardrobe page.
2. Once there, you can select a category (top, bottom, overtop) and color to filter the wardrobe and explore available items.

### Adding a Clothing Item

To add a new clothing item to the system:

1. Click on **Add Clothing Item** to go to the page for adding new items.
2. Input the details of the new clothing item, including category, color, and any other required information.

## Enjoy OutfitOracle

Explore these features to get the most out of OutfitOracle. We hope the application helps you effortlessly decide on outfits that suit your style and the weather conditions.



