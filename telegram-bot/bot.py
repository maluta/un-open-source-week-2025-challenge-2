import os
import asyncio
import logging
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get bot token from environment variable
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Validate that token is present
if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable is not set!")
    logger.error("Please set it in your .env file or as an environment variable")
    logger.error("Example: TELEGRAM_BOT_TOKEN=your_bot_token_here")
    raise ValueError("TELEGRAM_BOT_TOKEN is required but not found in environment variables")

# Directory where your PNG images are stored
IMAGES_DIR = Path('./images')
IMAGES_DIR.mkdir(exist_ok=True)

# Mapping of countries to their corresponding image files
COUNTRY_IMAGES = {
    'cambodia': 'cambodia_vulnerable_children.png',
    'camboja': 'cambodia_vulnerable_children.png',  # Portuguese for Cambodia
    'kenya': 'kenya_vulnerable_children.png'
}

# Response data for each country
COUNTRY_DATA = {
    'cambodia': {
        'children_count': '2.3 million',
        'storm_type': 'tropical cyclones and flooding',
        'additional_info': 'Children in rural areas are particularly vulnerable due to limited access to safe shelters and emergency services.'
    },
    'camboja': {
        'children_count': '2.3 million',
        'storm_type': 'tropical cyclones and flooding', 
        'additional_info': 'Children in rural areas are particularly vulnerable due to limited access to safe shelters and emergency services.'
    },
    'kenya': {
        'children_count': '1.8 million',
        'storm_type': 'severe weather events and droughts',
        'additional_info': 'Coastal regions and arid areas face the highest risk, with children in informal settlements being most vulnerable.'
    }
}

class VulnerableChildrenBot:
    def __init__(self):
        # Patterns to match the vulnerable children questions
        self.vulnerable_children_patterns = [
            r'how\s+many\s+vulnerable\s+children\s+will\s+be\s+hit\s+by\s+the\s+next\s+storm\s+in\s+(\w+)',
            r'vulnerable\s+children.*storm.*in\s+(\w+)',
            r'storm.*vulnerable\s+children.*(\w+)',
            r'children.*vulnerable.*storm.*(\w+)',
            r'(\w+).*vulnerable\s+children.*storm'
        ]
    
    def extract_country(self, text: str) -> Optional[str]:
        """Extract country name from the message"""
        text_lower = text.lower().strip()
        
        # Try each pattern
        for pattern in self.vulnerable_children_patterns:
            match = re.search(pattern, text_lower)
            if match:
                country = match.group(1).lower()
                # Normalize country names
                if country in ['cambodia', 'camboja']:
                    return 'cambodia' if country == 'cambodia' else 'camboja'
                elif country == 'kenya':
                    return 'kenya'
        
        # Direct country name check
        for country in COUNTRY_IMAGES.keys():
            if country in text_lower:
                return country
        
        return None
    
    def get_country_image_path(self, country: str) -> Optional[Path]:
        """Get the image path for a specific country"""
        if country in COUNTRY_IMAGES:
            image_path = IMAGES_DIR / COUNTRY_IMAGES[country]
            if image_path.exists():
                return image_path
            else:
                logger.warning(f"Image file not found: {image_path}")
                return None
        return None
    
    def get_country_response(self, country: str) -> str:
        """Get the text response for a specific country"""
        if country in COUNTRY_DATA:
            data = COUNTRY_DATA[country]
            response = f"""🚨 **Vulnerable Children Storm Impact Report**

📍 **Country**: {country.title()}
👶 **Vulnerable Children**: {data['children_count']}
🌪️ **Storm Type**: {data['storm_type']}

ℹ️ **Additional Information**: 
{data['additional_info']}

📊 The attached image shows detailed statistics and geographical distribution of vulnerable areas."""
            return response
        return "Data not available for this country."

# Initialize bot instance
vulnerable_bot = VulnerableChildrenBot()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_message = """
🤖 **Vulnerable Children Storm Impact Bot**

I can provide information about vulnerable children affected by storms in specific countries.

**Supported queries:**
• "How many vulnerable children will be hit by the next storm in Cambodia?"
• "How many vulnerable children will be hit by the next storm in Kenya?"
• "Vulnerable children storm impact in Camboja"

**Supported Countries:**
🇰🇭 Cambodia (Camboja)
🇰🇪 Kenya

Just ask me about vulnerable children and storms in these countries!
    """
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_message = """
📚 **Help - Vulnerable Children Storm Bot**

**How to use:**
Send me a message asking about vulnerable children and storms in supported countries.

**Example queries:**
• "How many vulnerable children will be hit by the next storm in Cambodia?"
• "Storm impact on vulnerable children in Kenya"
• "Vulnerable children Camboja storm"

**Supported Countries:**
🇰🇭 **Cambodia** (also: Camboja)
🇰🇪 **Kenya**

**What you'll get:**
• Statistical data about vulnerable children
• Storm impact information
• Detailed infographic (PNG image)

**Commands:**
/start - Start the bot
/help - Show this help
/countries - List supported countries
    """
    await update.message.reply_text(help_message, parse_mode='Markdown')

async def countries_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /countries command"""
    countries_message = """
🌍 **Supported Countries**

🇰🇭 **Cambodia** (Camboja)
• Image file: `cambodia_vulnerable_children.png`
• Data: Tropical cyclones and flooding impact

🇰🇪 **Kenya**  
• Image file: `kenya_vulnerable_children.png`
• Data: Severe weather and drought impact

**Note:** Make sure the corresponding PNG files are placed in the `./images/` directory.
    """
    await update.message.reply_text(countries_message, parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages"""
    text = update.message.text
    chat_id = update.effective_chat.id
    
    if not text:
        return
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
    
    try:
        # Extract country from the message
        country = vulnerable_bot.extract_country(text)
        
        if country:
            await handle_vulnerable_children_query(update, context, country)
        else:
            # Check if it's a general vulnerable children question
            if any(keyword in text.lower() for keyword in ['vulnerable', 'children', 'storm', 'hit']):
                guidance_message = """
❓ I understand you're asking about vulnerable children and storms, but I need to know the specific country.

**Please specify one of these countries:**
🇰🇭 Cambodia (or Camboja)
🇰🇪 Kenya

**Example:**
"How many vulnerable children will be hit by the next storm in Cambodia?"
                """
                await update.message.reply_text(guidance_message, parse_mode='Markdown')
            else:
                # General guidance
                guidance_message = """
🤖 I'm specialized in providing information about **vulnerable children affected by storms**.

**Try asking:**
• "How many vulnerable children will be hit by the next storm in Cambodia?"
• "Storm impact on vulnerable children in Kenya"

Use /help for more information!
                """
                await update.message.reply_text(guidance_message, parse_mode='Markdown')
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await update.message.reply_text("Sorry, I encountered an error processing your request. Please try again.")

async def handle_vulnerable_children_query(update: Update, context: ContextTypes.DEFAULT_TYPE, country: str):
    """Handle vulnerable children storm query for specific country"""
    chat_id = update.effective_chat.id
    
    try:
        # Get the image path for the country
        image_path = vulnerable_bot.get_country_image_path(country)
        
        if image_path and image_path.exists():
            # Get the response text
            response_text = vulnerable_bot.get_country_response(country)
            
            # Send the image with caption
            with open(image_path, 'rb') as f:
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=f,
                    caption=response_text,
                    parse_mode='Markdown'
                )
        else:
            # Image not found, send text response only
            error_message = f"""
❌ **Image Not Found**

I have data for {country.title()}, but the corresponding image file is missing.

**Expected file:** `{COUNTRY_IMAGES.get(country, 'unknown')}`
**Location:** `./images/` directory

{vulnerable_bot.get_country_response(country)}

Please ensure the image file is placed in the correct directory.
            """
            await update.message.reply_text(error_message, parse_mode='Markdown')
            
    except Exception as e:
        logger.error(f"Error handling vulnerable children query: {e}")
        await update.message.reply_text("❌ Error retrieving information. Please try again.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")

def main():
    """Start the bot"""
    # Log token status (without revealing the actual token)
    if BOT_TOKEN:
        masked_token = f"{BOT_TOKEN[:8]}...{BOT_TOKEN[-4:]}" if len(BOT_TOKEN) > 12 else "***"
        logger.info(f"Bot token loaded successfully (masked): {masked_token}")
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("countries", countries_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    # Check if image directory exists and warn about missing files
    logger.info("🤖 Vulnerable Children Storm Bot starting...")
    logger.info(f"Images directory: {IMAGES_DIR.absolute()}")
    
    for country, filename in COUNTRY_IMAGES.items():
        image_path = IMAGES_DIR / filename
        if image_path.exists():
            logger.info(f"✅ Found image for {country}: {filename}")
        else:
            logger.warning(f"❌ Missing image for {country}: {filename}")
    
    # Start the bot
    logger.info("Bot is ready to receive vulnerable children queries...")
    
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")

if __name__ == '__main__':
    main()