from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from gradio_client import Client, handle_file
import os
import tempfile
import logging
import base64

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)
logger = app.logger

try:
    client = Client("ohayonguy/PMRF", show_error=True)
except Exception as e:
    logger.error(f"Failed to initialize Gradio client: {str(e)}")
    raise

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
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_input:
                file.save(temp_input.name)
                logger.info(f"Saved uploaded file to temporary location: {temp_input.name}")
            
            logger.info("Starting photo restoration process")
            result = client.predict(
                handle_file(temp_input.name),
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
        
        except Exception as e:
            logger.error(f"Error during photo restoration: {str(e)}", exc_info=True)
            return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500
        
        finally:
            if 'temp_input' in locals():
                os.unlink(temp_input.name)
                logger.info(f"Deleted temporary input file: {temp_input.name}")
            if 'restored_image_path' in locals() and os.path.exists(restored_image_path):
                os.remove(restored_image_path)
                logger.info(f"Deleted restored image file: {restored_image_path}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))