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
# Note: These are example values - real data would come from CCRI-DRM models
COUNTRY_DATA = {
    'cambodia': {
        'children_count': '2.3 million',
        'storm_type': 'tropical cyclones and flooding',
        'additional_info': 'Rural children face higher risks due to limited access to early warning systems, safe shelters, and emergency services. Many schools lack storm-resistant structures.',
        'data_challenges': 'Current analysis combines weather forecasts (grid data) with child vulnerability indicators (administrative boundaries) and school locations (point data).',
        'unicef_action': 'UNICEF is developing automated systems to predict which communities need support before storms hit.'
    },
    'camboja': {
        'children_count': '2.3 million',
        'storm_type': 'tropical cyclones and flooding', 
        'additional_info': 'Rural children face higher risks due to limited access to early warning systems, safe shelters, and emergency services. Many schools lack storm-resistant structures.',
        'data_challenges': 'Current analysis combines weather forecasts (grid data) with child vulnerability indicators (administrative boundaries) and school locations (point data).',
        'unicef_action': 'UNICEF is developing automated systems to predict which communities need support before storms hit.'
    },
    'kenya': {
        'children_count': '1.8 million',
        'storm_type': 'severe floods and droughts',
        'additional_info': 'Children in informal settlements and arid regions are most at risk. Limited access to clean water and sanitation increases disease risk after flooding.',
        'data_challenges': 'Matching real-time weather data with locations of schools, health centers, and vulnerable populations remains a manual process.',
        'unicef_action': 'Working on automated early warning systems that can identify at-risk children 48-72 hours before extreme weather events.'
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
            response = f"""🚨 **Children at Risk: Storm Impact Analysis**

📍 **Country**: {country.title()}
👶 **Vulnerable Children**: {data['children_count']}
🌪️ **Main Hazards**: {data['storm_type']}

**Why are children at risk?**
{data['additional_info']}

**The Data Challenge:**
{data['data_challenges']}

**What UNICEF is doing:**
{data['unicef_action']}

⚠️ **Note**: This is a demonstration using example data. Real impact assessments require complex analysis combining weather forecasts, population data, and infrastructure information - exactly what UNICEF's CCRI-DRM system aims to automate.

📊 The attached visualization shows potential impact zones and vulnerable populations."""
            return response
        return "Data not available for this country."

# Initialize bot instance
vulnerable_bot = VulnerableChildrenBot()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_message = """
🤖 **UNICEF Storm Impact Demo Bot**

This bot demonstrates how UNICEF analyzes storm impacts on vulnerable children by combining weather data with population information.

**The Real Challenge:**
Predicting which children will be affected by storms requires combining:
• Weather forecasts (grid/raster data)
• Child vulnerability indicators (administrative boundaries)
• School/health center locations (point data)

Currently, this process is manual and time-consuming. UNICEF is building automated systems to provide faster, more accurate warnings.

**Try these queries:**
• "How many vulnerable children will be hit by the next storm in Cambodia?"
• "Storm impact on children in Kenya"

**Available Countries:** 🇰🇭 Cambodia | 🇰🇪 Kenya

⚠️ **Disclaimer**: This bot uses example data for demonstration. Real assessments require complex spatial analysis.
    """
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_message = """
📚 **About This Demo**

**What this bot simulates:**
UNICEF's efforts to predict storm impacts on children by analyzing:
• Where storms will hit (weather data)
• Where vulnerable children live (demographic data)
• What resources are available (schools, health centers)

**The Technical Challenge:**
These datasets come in different formats:
• Weather = Grid/raster format
• Child data = Administrative boundaries
• Infrastructure = GPS points

Making them work together is like solving a "geo-puzzle" - which is what UNICEF's CCRI-DRM and Giga Spatial projects aim to automate.

**Example queries:**
• "How many vulnerable children will be hit by the next storm in Cambodia?"
• "Storm impact on children in Kenya"

**Commands:**
/start - About this demo
/help - Technical details
/countries - Available data
/challenge - Learn about the real problem

💡 **Note**: This is a hackathon prototype demonstrating the concept. Real implementations would pull live weather data and actual vulnerability indicators.
    """
    await update.message.reply_text(help_message, parse_mode='Markdown')

async def challenge_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /challenge command - explains the real problem UNICEF faces"""
    challenge_message = """
🎯 **The Real Challenge UNICEF Faces**

**The Problem:**
When a hurricane approaches, UNICEF needs to know:
• Which children will be affected?
• Where are they located?
• What resources do they have access to?

But the data comes in incompatible formats:
📊 Weather forecasts = Grid cells (1km x 1km squares)
📍 School locations = GPS coordinates (lat/lon points)
🗺️ Poverty data = Administrative boundaries (districts)

**Current Reality:**
• Analysts manually overlay these datasets
• Takes 6-12 hours per country
• By then, the storm is closer
• Errors from manual processing

**What's Needed:**
An automated system that can:
1. Pull weather forecasts automatically
2. Match them with child vulnerability data
3. Identify at-risk schools/health centers
4. Generate alerts in <30 minutes

**Why It Matters:**
• 48-hour warning = Time to evacuate schools
• 24-hour warning = Emergency supplies positioned
• 12-hour warning = Too late for many actions

This bot demonstrates the concept. The real solution needs to handle terabytes of satellite data, millions of population records, and update every 6 hours.

Learn more: github.com/unicef/giga-spatial
    """
    await update.message.reply_text(challenge_message, parse_mode='Markdown')

async def countries_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /countries command"""
    countries_message = """
🌍 **Demo Data Available**

🇰🇭 **Cambodia**
• Population: 16.7 million (40% children)
• Key risks: Monsoons, tropical storms, Mekong flooding
• Data gaps: Rural areas lack real-time monitoring
• CCRI-DRM Status: Dashboard active

🇰🇪 **Kenya**  
• Population: 53 million (42% children)
• Key risks: Floods, droughts, extreme temperatures
• Data gaps: Informal settlements poorly mapped
• CCRI-DRM Status: Dashboard active

**What we need to automate:**
1. Download weather forecasts (ECMWF, GFS)
2. Match with child population data
3. Identify schools/health centers at risk
4. Generate alerts 48-72 hours ahead

**Current process:** Manual, takes 6-12 hours
**Goal:** Automated, under 30 minutes

📊 PNG visualizations in `./images/` simulate the output of automated analysis.
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
🤖 This is a demo bot showing how UNICEF could provide storm impact data.

**Try asking about:**
• "How many vulnerable children will be hit by the next storm in Cambodia?"
• "Storm impact on children in Kenya"

**Learn more:**
/help - Technical details
/challenge - The real problem UNICEF faces
/countries - Available demo data

💡 **Reality check**: A real system would pull live weather data, use actual population statistics, and update every 6 hours. This demo uses static example data to illustrate the concept.
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
    application.add_handler(CommandHandler("challenge", challenge_command))
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