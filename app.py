from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import List
import io
from PIL import Image
import logging

app = FastAPI(title="Image Stitching API", description="Automated panorama creation from multiple images")

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageStitcher:
    def __init__(self):
        # Initialize ORB detector (faster than SIFT/SURF and patent-free)
        self.detector = cv2.ORB_create(nfeatures=5000)
        # FLANN matcher for fast matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def preprocess_image(self, img):
        """Preprocess image for better stitching results"""
        # Resize if too large (maintain aspect ratio)
        height, width = img.shape[:2]
        if width > 1500:
            scale = 1500 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Apply slight Gaussian blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img
    
    def detect_and_match_features(self, img1, img2):
        """Detect features and find matches between two images"""
        # Convert to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            return None, None, None
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return kp1, kp2, good_matches
    
    def find_homography_ransac(self, kp1, kp2, matches):
        """Find homography matrix using RANSAC"""
        if len(matches) < 4:
            return None
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography with RANSAC
        homography, mask = cv2.findHomography(
            src_pts, dst_pts, 
            cv2.RANSAC, 
            ransacReprojThreshold=5.0,
            confidence=0.99,
            maxIters=5000
        )
        
        return homography
    
    def warp_and_blend_images(self, img1, img2, homography):
        """Warp second image and blend with first image"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get corners of second image
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners2_transformed = cv2.perspectiveTransform(corners2, homography)
        
        # Find the bounding box for the final panorama
        all_corners = np.concatenate([
            np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
            corners2_transformed
        ], axis=0)
        
        x_coords = all_corners[:, 0, 0]
        y_coords = all_corners[:, 0, 1]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Translation matrix to handle negative coordinates
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        
        # Final canvas size
        canvas_width = x_max - x_min
        canvas_height = y_max - y_min
        
        # Warp the second image
        homography_translated = translation @ homography
        img2_warped = cv2.warpPerspective(img2, homography_translated, (canvas_width, canvas_height))
        
        # Place first image on canvas
        img1_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        y_offset, x_offset = -y_min, -x_min
        img1_canvas[y_offset:y_offset+h1, x_offset:x_offset+w1] = img1
        
        # Create masks for blending
        mask1 = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        mask1[y_offset:y_offset+h1, x_offset:x_offset+w1] = 1.0
        
        mask2 = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        mask2[img2_warped.sum(axis=2) > 0] = 1.0
        
        # Multi-band blending in overlap region
        overlap_mask = (mask1 > 0) & (mask2 > 0)
        
        if np.any(overlap_mask):
            # Distance transform for smooth blending weights
            dist1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)
            
            # Normalize distances for blending weights
            total_dist = dist1 + dist2
            total_dist[total_dist == 0] = 1  # Avoid division by zero
            
            weight1 = dist1 / total_dist
            weight2 = dist2 / total_dist
            
            # Apply blending
            result = np.zeros_like(img1_canvas, dtype=np.float32)
            
            for c in range(3):
                result[:, :, c] = (
                    weight1 * img1_canvas[:, :, c].astype(np.float32) +
                    weight2 * img2_warped[:, :, c].astype(np.float32)
                )
            
            # Handle non-overlap regions
            result[mask1 > 0] = np.where(
                overlap_mask[mask1 > 0, np.newaxis],
                result[mask1 > 0],
                img1_canvas[mask1 > 0].astype(np.float32)
            )
            result[mask2 > 0] = np.where(
                overlap_mask[mask2 > 0, np.newaxis],
                result[mask2 > 0],
                img2_warped[mask2 > 0].astype(np.float32)
            )
            
        else:
            # No overlap, simple combination
            result = img1_canvas.astype(np.float32)
            result[mask2 > 0] = img2_warped[mask2 > 0].astype(np.float32)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def stitch_images(self, images):
        """Main stitching function for multiple images"""
        if len(images) < 2:
            raise ValueError("At least 2 images are required for stitching")
        
        # Preprocess all images
        processed_images = [self.preprocess_image(img) for img in images]
        
        # Start with the first image as base
        result = processed_images[0]
        
        # Progressively stitch each subsequent image
        for i in range(1, len(processed_images)):
            logger.info(f"Stitching image {i+1}/{len(processed_images)}")
            
            current_img = processed_images[i]
            
            # Find features and matches
            kp1, kp2, matches = self.detect_and_match_features(result, current_img)
            
            if matches is None or len(matches) < 10:
                logger.warning(f"Insufficient matches found between images. Found: {len(matches) if matches else 0}")
                continue
            
            # Find homography
            homography = self.find_homography_ransac(kp1, kp2, matches)
            
            if homography is None:
                logger.warning(f"Could not find valid homography for image {i+1}")
                continue
            
            # Warp and blend
            result = self.warp_and_blend_images(result, current_img, homography)
            
            logger.info(f"Successfully stitched image {i+1}")
        
        return result

# Global stitcher instance
stitcher = ImageStitcher()

@app.post("/stitch")
async def stitch_images(files: List[UploadFile] = File(...)):
    """
    Stitch multiple images into a panorama
    """
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="At least 2 images are required")
        
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
        
        # Load and validate images
        images = []
        for file in files:
            # Validate file type
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
            
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail=f"Could not decode image {file.filename}")
            
            images.append(img)
        
        logger.info(f"Starting stitching process with {len(images)} images")
        
        # Perform stitching
        result = stitcher.stitch_images(images)
        
        # Convert result to bytes
        success, encoded_img = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode result image")
        
        # Return as streaming response
        img_bytes = io.BytesIO(encoded_img.tobytes())
        img_bytes.seek(0)
        
        logger.info("Stitching completed successfully")
        
        return StreamingResponse(
            img_bytes, 
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=panorama.jpg"}
        )
        
    except Exception as e:
        logger.error(f"Error during stitching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stitching failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Image stitching service is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Image Stitching API",
        "version": "1.0.0",
        "endpoints": {
            "/stitch": "POST - Upload multiple images for stitching",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)