import numpy as np
import random
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ResponseTemplates:
    """Manages response templates for different customer support scenarios"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.custom_templates = {}
    
    def _initialize_templates(self) -> Dict:
        """Initialize default response templates"""
        return {
            'escalation': {
                'urgent': [
                    "I understand this is urgent. I'm escalating this to our priority support team who will contact you within 15 minutes.",
                    "Thank you for bringing this urgent matter to our attention. A specialist will reach out to you immediately.",
                    "I see this requires immediate attention. Let me connect you with our emergency support team right away."
                ],
                'standard': [
                    "I'd like to help resolve this for you. For security purposes, please send us a private message.",
                    "To better assist you with your account details, please click 'Message' at the top of your profile.",
                    "Let me get you connected with the right specialist. Please send us a private message to continue."
                ]
            },
            'acknowledgment': {
                'frustrated': [
                    "I sincerely apologize for the inconvenience you've experienced. Let me personally ensure this gets resolved.",
                    "I understand your frustration and I'm here to make this right. Thank you for your patience.",
                    "I'm sorry this hasn't been resolved yet. Let me take ownership of this issue right now."
                ],
                'neutral': [
                    "Thank you for contacting us. I'm here to help resolve this matter.",
                    "I appreciate you reaching out. Let me assist you with this right away.",
                    "Thank you for bringing this to our attention. I'll help you get this sorted out."
                ]
            },
            'instruction': [
                "Here's how to resolve this: [specific steps will be inserted]",
                "Please try these steps: [detailed instructions will be provided]",
                "To fix this issue, please follow these instructions: [steps will be customized]"
            ],
            'followup': [
                "I wanted to follow up on your previous message. Were you able to resolve the issue?",
                "Just checking in - did the solution I provided work for you?",
                "How did everything work out? Please let me know if you need any additional assistance."
            ],
            'general': [
                "Thank you for contacting us. How can I assist you today?",
                "I'll be happy to help with that. Could you provide more details?",
                "Thanks for reaching out. Let me look into this for you."
            ]
        }
    
    def get_template(self, 
                   category: str, 
                   urgency_score: float = 0, 
                   frustration_score: float = 0) -> str:
        """
        Get the most appropriate response template based on:
        - Message category
        - Urgency level
        - Frustration level
        
        Args:
            category: Predicted response category (escalation, acknowledgment, etc.)
            urgency_score: Calculated urgency score (0-10)
            frustration_score: Calculated frustration score (0-10)
            
        Returns:
            str: Appropriate response template
        """
        try:
            # First check custom templates
            custom_key = f"{category}_{urgency_score}_{frustration_score}"
            if custom_key in self.custom_templates:
                return random.choice(self.custom_templates[custom_key])
            
            # Use default templates
            if category == 'escalation':
                if urgency_score > 5:
                    return np.random.choice(self.templates['escalation']['urgent'])
                return np.random.choice(self.templates['escalation']['standard'])
            
            elif category == 'acknowledgment':
                if frustration_score > 3:
                    return np.random.choice(self.templates['acknowledgment']['frustrated'])
                return np.random.choice(self.templates['acknowledgment']['neutral'])
            
            elif category in self.templates:
                return np.random.choice(self.templates[category])
            
            return "Thank you for contacting us. How can I assist you today?"
        
        except Exception as e:
            logger.error(f"Error selecting template: {e}")
            return "Thank you for your message. We'll get back to you shortly."
    
    def add_custom_template(self, 
                          category: str, 
                          template: str,
                          urgency_threshold: Optional[float] = None,
                          frustration_threshold: Optional[float] = None):
        """
        Add a custom response template for specific conditions
        
        Args:
            category: Response category
            template: The response template text
            urgency_threshold: Minimum urgency score for this template
            frustration_threshold: Minimum frustration score for this template
        """
        try:
            key = category
            if urgency_threshold is not None:
                key += f"_{urgency_threshold}"
            if frustration_threshold is not None:
                key += f"_{frustration_threshold}"
            
            if key not in self.custom_templates:
                self.custom_templates[key] = []
            
            self.custom_templates[key].append(template)
            logger.info(f"Added custom template for {key}")
            
        except Exception as e:
            logger.error(f"Error adding custom template: {e}")
    
    def save_templates(self, filepath: str):
        """Save all templates to a JSON file"""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump({
                    'default_templates': self.templates,
                    'custom_templates': self.custom_templates
                }, f, indent=2)
            logger.info(f"Saved templates to {filepath}")
        except Exception as e:
            logger.error(f"Error saving templates: {e}")
    
    def load_templates(self, filepath: str):
        """Load templates from a JSON file"""
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.templates = data.get('default_templates', self.templates)
                self.custom_templates = data.get('custom_templates', {})
            logger.info(f"Loaded templates from {filepath}")
        except Exception as e:
            logger.error(f"Error loading templates: {e}")

    def get_all_templates(self) -> Dict:
        """Return all templates (default + custom) for inspection"""
        return {
            'default': self.templates,
            'custom': self.custom_templates
        }