from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError
import os
import tempfile
import logging
import base64
import traceback

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = app.logger

# Initialize Gradio client
try:
    client = Client("ohayonguy/PMRF")
    logger.info("Gradio client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gradio client: {str(e)}")
    logger.error(traceback.format_exc())
    # Don't raise the exception, let the application continue

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/restore', methods=['POST'])
def restore_photo():
    logger.info("Received photo restoration request")
   
    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({'success': False, 'error': 'No file part'}), 400
   
    file = request.files['file']
   
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'success': False, 'error': 'No selected file'}), 400
   
    if file:
        temp_input = None
        restored_image_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp_input = temp.name
                file.save(temp_input)
                logger.info(f"Saved uploaded file to temporary location: {temp_input}")
           
            logger.info("Starting photo restoration process")
            result = client.predict(
                handle_file(temp_input),
                True,  # randomize_seed
                False,  # aligned
                1,  # scale
                25,  # num_flow_steps
                42,  # seed
                api_name="/predict"
            )
            logger.info("Photo restoration process completed")
            restored_image_path, _ = result
            if restored_image_path and os.path.exists(restored_image_path):
                logger.info(f"Reading restored image: {restored_image_path}")
                with open(restored_image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return jsonify({'success': True, 'image': encoded_string})
            else:
                logger.error("Failed to retrieve the restored image")
                return jsonify({'success': False, 'error': 'Failed to retrieve the restored image'}), 500
        
        except AppError as e:
            logger.error(f"Gradio AppError: {str(e)}")
            return jsonify({'success': False, 'error': 'The photo restoration service is currently unavailable. Please try again later or contact support.'}), 503
       
        except Exception as e:
            logger.error(f"Error during photo restoration: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'success': False, 'error': 'An internal server error occurred. Please try again later.'}), 500
       
        finally:
            if temp_input and os.path.exists(temp_input):
                os.unlink(temp_input)
                logger.info(f"Deleted temporary input file: {temp_input}")
            if restored_image_path and os.path.exists(restored_image_path):
                os.remove(restored_image_path)
                logger.info(f"Deleted restored image file: {restored_image_path}")

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal Server Error: {error}")
    return jsonify({'success': False, 'error': 'An unexpected error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)