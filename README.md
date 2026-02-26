# SNSE - AI Virtual Fitting Room 🧥✨

SNSE is a Chrome Extension that transforms static e-commerce product pages into an interactive virtual fitting room. By leveraging Google's advanced Multimodal AI, SNSE allows users to instantly visualize how specific online clothing items look as part of a complete, photorealistic outfit.

**⚠️ CURRENT LIMITATION:** This MVP is currently configured to scrape and process DOM elements **exclusively on [abercrombie.com](https://www.abercrombie.com/)**. Support for additional retailers is planned for future updates.

## 🚀 Features

* **Real-Time Outfit Generation:** Generates high-fidelity, photorealistic fashion images of a complete outfit using the item you are currently viewing.
* **Smart Asset Caching:** Utilizes the Chrome Storage API to remember your generated outfits. If you navigate away and come back to a product, your custom look is instantly loaded from local memory without requiring a redundant API call.
* **Dynamic Image Processing:** Features a custom "on-the-fly" image processing pipeline that automatically sanitizes DOM image URLs and converts them to Base64 payloads to seamlessly bypass CORS and MIME-type constraints.
* **Virtual Closet:** Save individual pieces ("Pickups") and full outfits to your local extension storage to build your wardrobe across browsing sessions.

## 🛠️ Technical Stack

* **Frontend:** Vanilla JavaScript, HTML5, CSS3
* **Environment:** Google Chrome Extension (Manifest V3)
* **AI Integration:** Google Gemini 3 Pro Image Preview API (REST)
* **Data Management:** `chrome.storage.local`

## ⚙️ Installation & Setup

Because this extension uses an external AI model, **you must provide your own Google Gemini API key** to use the generation features.

1. **Get an API Key:** Obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).
2. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/snse-extension.git](https://github.com/yourusername/snse-extension.git)
