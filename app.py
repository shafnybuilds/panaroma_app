from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import List, Tuple, Optional
import io
from PIL import Image
import logging
from dataclasses import dataclass
import math

app = FastAPI(title="Advanced Image Stitching API", description="Professional panorama creation with state-of-the-art algorithms")

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

@dataclass
class ImageMatch:
    """Represents a match between two images"""
    img1_idx: int
    img2_idx: int
    matches: List
    homography: np.ndarray
    confidence: float

class ProfessionalImageStitcher:
    def __init__(self):
        # Initialize multiple feature detectors for robustness
        self.detectors = []
        
        # Try SIFT first (best quality)
        try:
            self.detectors.append(('SIFT', cv2.SIFT_create(
                nfeatures=8000,  # Reduced for better performance
                contrastThreshold=0.01,  # More sensitive
                edgeThreshold=15,  # More relaxed
                sigma=1.6
            )))
            logger.info("SIFT detector initialized")
        except Exception as e:
            logger.warning(f"SIFT not available: {e}")
        
        # Try AKAZE (good alternative)
        try:
            self.detectors.append(('AKAZE', cv2.AKAZE_create(
                threshold=0.0001,  # More sensitive
                nOctaves=4,
                nOctaveLayers=4
            )))
            logger.info("AKAZE detector initialized")
        except Exception as e:
            logger.warning(f"AKAZE not available: {e}")
        
        # Fallback to ORB
        try:
            self.detectors.append(('ORB', cv2.ORB_create(
                nfeatures=8000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,  # More relaxed
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=10  # More sensitive
            )))
            logger.info("ORB detector initialized")
        except Exception as e:
            logger.warning(f"ORB not available: {e}")
        
        if not self.detectors:
            raise RuntimeError("No feature detectors available")
        
        # Matcher configuration
        self.matcher = cv2.BFMatcher()
        
        # More relaxed stitching parameters
        self.min_match_count = 15  # Reduced from 50
        self.ransac_threshold = 6.0  # Increased from 4.0
        self.confidence_threshold = 0.95  # Reduced from 0.99
        
    def preprocess_image(self, img: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
        """Enhanced preprocessing with intelligent resizing and enhancement"""
        if img is None:
            raise ValueError("Image is None")
            
        height, width = img.shape[:2]
        
        # More conservative resizing to preserve detail
        if max(height, width) > max_dimension:
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Enhanced preprocessing for better feature detection
        # 1. Convert to LAB color space for better contrast
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 2. Apply CLAHE to improve contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Reduced clip limit
        l = clahe.apply(l)
        
        # 3. Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 4. Apply mild Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        
        return enhanced
    
    def detect_and_compute_features(self, img: np.ndarray) -> Tuple[List, np.ndarray, str]:
        """Detect features using multiple detectors and combine results"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        all_keypoints = []
        all_descriptors = []
        detector_types = []
        
        for detector_name, detector in self.detectors:
            try:
                keypoints, descriptors = detector.detectAndCompute(gray, None)
                
                if descriptors is not None and len(keypoints) > 10:  # Minimum threshold
                    all_keypoints.extend(keypoints)
                    if len(all_descriptors) == 0:
                        all_descriptors = descriptors
                    else:
                        # Only combine if same descriptor type
                        if descriptors.shape[1] == all_descriptors.shape[1]:
                            all_descriptors = np.vstack([all_descriptors, descriptors])
                        elif len(all_descriptors) < len(descriptors):
                            # Use the detector with more features
                            all_keypoints = keypoints
                            all_descriptors = descriptors
                            detector_types = [detector_name]
                            continue
                    
                    detector_types.append(detector_name)
                    logger.info(f"Detected {len(keypoints)} features using {detector_name}")
                    
            except Exception as e:
                logger.warning(f"{detector_name} detection failed: {e}")
                continue
        
        if len(all_keypoints) == 0:
            logger.error("No features detected with any detector")
            return [], None, "None"
        
        primary_detector = detector_types[0] if detector_types else "Unknown"
        logger.info(f"Total features detected: {len(all_keypoints)} using {', '.join(detector_types)}")
        return all_keypoints, all_descriptors, primary_detector
    
    def match_features(self, des1: np.ndarray, des2: np.ndarray, detector_type: str) -> List:
        """More flexible feature matching with multiple strategies"""
        if des1 is None or des2 is None:
            return []
        
        try:
            # Try multiple matching strategies
            good_matches = []
            
            # Strategy 1: Use appropriate matcher based on descriptor type
            if detector_type == 'SIFT' and des1.dtype == np.float32:
                # For SIFT, use FLANN matcher
                try:
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)
                    
                    # Apply Lowe's ratio test
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:  # More relaxed ratio
                                good_matches.append(m)
                except Exception as e:
                    logger.warning(f"FLANN matching failed: {e}")
                    good_matches = []
            
            # Strategy 2: Brute force matcher (fallback or for ORB/AKAZE)
            if len(good_matches) < 10:
                try:
                    if detector_type in ['ORB', 'AKAZE']:
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                    else:
                        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                    
                    matches = matcher.knnMatch(des1, des2, k=2)
                    
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:  # More relaxed ratio
                                good_matches.append(m)
                                
                except Exception as e:
                    logger.warning(f"BF matching failed: {e}")
                    good_matches = []
            
            # Strategy 3: Cross-check matching if still not enough matches
            if len(good_matches) < 10:
                try:
                    if detector_type in ['ORB', 'AKAZE']:
                        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    else:
                        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                    
                    matches = matcher.match(des1, des2)
                    good_matches = sorted(matches, key=lambda x: x.distance)[:100]  # Take best 100
                    
                except Exception as e:
                    logger.warning(f"Cross-check matching failed: {e}")
            
            # Additional filtering based on distance if we have enough matches
            if len(good_matches) > 20:
                distances = [m.distance for m in good_matches]
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                threshold = mean_dist + 2.0 * std_dist  # More relaxed
                good_matches = [m for m in good_matches if m.distance <= threshold]
            
            logger.info(f"Found {len(good_matches)} good matches using {detector_type}")
            return good_matches
            
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            return []
    
    def estimate_homography(self, kp1: List, kp2: List, matches: List) -> Tuple[Optional[np.ndarray], float]:
        """More flexible homography estimation"""
        if len(matches) < self.min_match_count:
            logger.warning(f"Insufficient matches: {len(matches)} < {self.min_match_count}")
            return None, 0.0
        
        # Extract point correspondences
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Try multiple robust estimation methods with more relaxed parameters
        methods = [
            ('RANSAC', cv2.RANSAC, self.ransac_threshold, self.confidence_threshold, 3000),
            ('USAC_MAGSAC', cv2.USAC_MAGSAC, self.ransac_threshold, self.confidence_threshold, 5000),
            ('USAC_DEFAULT', cv2.USAC_DEFAULT, self.ransac_threshold * 1.5, self.confidence_threshold - 0.05, 3000),
            ('LMEDS', cv2.LMEDS, 0, self.confidence_threshold - 0.1, 2000)
        ]
        
        best_homography = None
        best_confidence = 0.0
        best_method = None
        
        for method_name, method, threshold, confidence, max_iters in methods:
            try:
                if method == cv2.LMEDS:
                    H, mask = cv2.findHomography(
                        src_pts, dst_pts,
                        method=method,
                        confidence=confidence,
                        maxIters=max_iters
                    )
                else:
                    H, mask = cv2.findHomography(
                        src_pts, dst_pts,
                        method=method,
                        ransacReprojThreshold=threshold,
                        confidence=confidence,
                        maxIters=max_iters
                    )
                
                if H is not None and mask is not None:
                    inlier_ratio = np.sum(mask) / len(mask)
                    
                    # More relaxed quality checks
                    if (inlier_ratio > 0.25 and  # Reduced from 0.4
                        self.validate_homography(H) and
                        inlier_ratio > best_confidence):
                        
                        best_homography = H
                        best_confidence = inlier_ratio
                        best_method = method_name
                        
            except Exception as e:
                logger.warning(f"Homography estimation with {method_name} failed: {e}")
                continue
        
        if best_homography is not None:
            logger.info(f"Best homography found using {best_method} with {best_confidence:.2%} inliers")
        else:
            logger.warning("No valid homography found")
            
        return best_homography, best_confidence
    
    def validate_homography(self, H: np.ndarray) -> bool:
        """More relaxed homography validation"""
        if H is None:
            return False
        
        try:
            # Check if matrix is well-conditioned
            det = np.linalg.det(H)
            if abs(det) < 1e-8:  # More relaxed
                return False
            
            # Check for reasonable scaling and rotation
            upper_left = H[:2, :2]
            scales = np.sqrt(np.sum(upper_left**2, axis=0))
            
            # More relaxed scaling limits (0.05x to 20x)
            if np.any(scales < 0.05) or np.any(scales > 20.0):
                return False
            
            # More relaxed perspective distortion check
            perspective = H[2, :2]
            if np.any(np.abs(perspective) > 0.05):  # Increased from 0.01
                return False
            
            return True
            
        except Exception:
            return False
    
    def find_best_matches(self, images: List[np.ndarray]) -> Tuple[List[ImageMatch], List]:
        """Find the best matches between all image pairs with better strategy"""
        n = len(images)
        image_features = []
        
        # Extract features from all images
        for i, img in enumerate(images):
            processed_img = self.preprocess_image(img)
            kp, des, detector_type = self.detect_and_compute_features(processed_img)
            image_features.append((processed_img, kp, des, detector_type))
            
            if des is None or len(kp) < 10:
                logger.warning(f"Image {i} has very few features ({len(kp) if kp else 0})")
        
        # Find matches between all pairs
        matches = []
        for i in range(n):
            for j in range(i + 1, n):
                img1, kp1, des1, det1 = image_features[i]
                img2, kp2, des2, det2 = image_features[j]
                
                if des1 is None or des2 is None:
                    logger.warning(f"Skipping pair {i}-{j}: missing descriptors")
                    continue
                
                # Use the primary detector type for matching
                detector_type = det1 if det1 != "Unknown" else det2
                
                feature_matches = self.match_features(des1, des2, detector_type)
                
                if len(feature_matches) >= self.min_match_count:
                    H, confidence = self.estimate_homography(kp1, kp2, feature_matches)
                    
                    if H is not None:
                        matches.append(ImageMatch(i, j, feature_matches, H, confidence))
                        logger.info(f"Valid match found between images {i} and {j}: "
                                  f"{len(feature_matches)} features, {confidence:.2%} confidence")
                else:
                    logger.info(f"Insufficient matches between images {i} and {j}: {len(feature_matches)}")
        
        # Sort matches by confidence (best first)
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        if not matches:
            # Try with even more relaxed parameters
            logger.warning("No matches found with standard parameters, trying relaxed matching...")
            self.min_match_count = 8  # Very relaxed
            self.ransac_threshold = 10.0
            
            for i in range(n):
                for j in range(i + 1, n):
                    img1, kp1, des1, det1 = image_features[i]
                    img2, kp2, des2, det2 = image_features[j]
                    
                    if des1 is None or des2 is None:
                        continue
                    
                    detector_type = det1 if det1 != "Unknown" else det2
                    feature_matches = self.match_features(des1, des2, detector_type)
                    
                    if len(feature_matches) >= self.min_match_count:
                        H, confidence = self.estimate_homography(kp1, kp2, feature_matches)
                        
                        if H is not None:
                            matches.append(ImageMatch(i, j, feature_matches, H, confidence))
                            logger.info(f"Relaxed match found between images {i} and {j}: "
                                      f"{len(feature_matches)} features, {confidence:.2%} confidence")
            
            matches.sort(key=lambda x: x.confidence, reverse=True)
        
        return matches, image_features
    
    def stitch_pair(self, img1: np.ndarray, img2: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Stitch two images using homography with advanced blending"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Calculate the output canvas size
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners2_transformed = cv2.perspectiveTransform(corners2, H)
        
        all_corners = np.concatenate([corners1, corners2_transformed], axis=0)
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
        
        # Create translation matrix to handle negative coordinates
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        
        # Canvas dimensions
        canvas_width = x_max - x_min
        canvas_height = y_max - y_min
        
        # Warp the second image
        H_translated = translation @ H
        img2_warped = cv2.warpPerspective(img2, H_translated, (canvas_width, canvas_height))
        
        # Place the first image on canvas
        img1_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        x_offset, y_offset = -x_min, -y_min
        img1_canvas[y_offset:y_offset+h1, x_offset:x_offset+w1] = img1
        
        # Create masks for blending
        mask1 = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        mask1[y_offset:y_offset+h1, x_offset:x_offset+w1] = 1.0
        
        mask2 = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        mask2[np.any(img2_warped != 0, axis=2)] = 1.0
        
        # Find overlap region for advanced blending
        overlap_mask = (mask1 > 0) & (mask2 > 0)
        
        if np.any(overlap_mask):
            # Use feather blending for better performance
            result = self.feather_blend(img1_canvas, img2_warped, mask1, mask2)
        else:
            # Simple combination for non-overlapping regions
            result = np.where(mask2[..., np.newaxis] > 0, img2_warped, img1_canvas)
        
        return result
    
    def feather_blend(self, img1: np.ndarray, img2: np.ndarray, 
                     mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """Faster feather blending alternative to multi-band blending"""
        
        # Create distance transforms for smooth blending
        mask1_dist = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5)
        mask2_dist = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5)
        
        # Normalize distance transforms
        total_dist = mask1_dist + mask2_dist
        total_dist[total_dist == 0] = 1  # Avoid division by zero
        
        weight1 = mask1_dist / total_dist
        weight2 = mask2_dist / total_dist
        
        # Apply weights
        weight1_3ch = cv2.merge([weight1, weight1, weight1])
        weight2_3ch = cv2.merge([weight2, weight2, weight2])
        
        # Blend images
        result = (img1.astype(np.float32) * weight1_3ch + 
                 img2.astype(np.float32) * weight2_3ch)
        
        # Handle non-overlapping regions
        overlap = (mask1 > 0) & (mask2 > 0)
        result[~overlap & (mask1 > 0)] = img1[~overlap & (mask1 > 0)]
        result[~overlap & (mask2 > 0)] = img2[~overlap & (mask2 > 0)]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def stitch_images(self, images: List[np.ndarray]) -> np.ndarray:
        """Main stitching function using graph-based approach"""
        if len(images) < 2:
            raise ValueError("At least 2 images required for stitching")
        
        logger.info(f"Starting panorama creation with {len(images)} images")
        
        # Find all valid matches
        matches, image_features = self.find_best_matches(images)
        
        if not matches:
            raise ValueError("No valid matches found between images. "
                           "Try images with more overlap, similar lighting, or distinctive features. "
                           "Consider using images from the same camera/session.")
        
        # Build stitching sequence using the strongest matches
        used_images = set()
        result_img = None
        
        # Start with the best match
        best_match = matches[0]
        img1_idx, img2_idx = best_match.img1_idx, best_match.img2_idx
        
        logger.info(f"Starting with images {img1_idx} and {img2_idx} "
                   f"(confidence: {best_match.confidence:.2%})")
        
        img1 = image_features[img1_idx][0]  # preprocessed image
        img2 = image_features[img2_idx][0]  # preprocessed image
        
        result_img = self.stitch_pair(img1, img2, best_match.homography)
        used_images.add(img1_idx)
        used_images.add(img2_idx)
        
        # Add remaining images
        while len(used_images) < len(images):
            best_next_match = None
            best_confidence = 0
            
            # Find the best match that includes an unused image
            for match in matches:
                if (match.img1_idx in used_images) != (match.img2_idx in used_images):
                    if match.confidence > best_confidence:
                        best_next_match = match
                        best_confidence = match.confidence
            
            if best_next_match is None:
                logger.warning("Cannot find more matches to extend panorama")
                break
            
            # Determine which image to add
            if best_next_match.img1_idx in used_images:
                new_img_idx = best_next_match.img2_idx
                H = best_next_match.homography
            else:
                new_img_idx = best_next_match.img1_idx
                H = np.linalg.inv(best_next_match.homography)  # Invert homography
            
            logger.info(f"Adding image {new_img_idx} (confidence: {best_confidence:.2%})")
            
            new_img = image_features[new_img_idx][0]
            result_img = self.stitch_pair(result_img, new_img, H)
            used_images.add(new_img_idx)
        
        # Final post-processing
        result_img = self.post_process(result_img)
        
        logger.info(f"Panorama completed using {len(used_images)}/{len(images)} images")
        return result_img
    
    def post_process(self, img: np.ndarray) -> np.ndarray:
        """Post-process the final panorama"""
        # Crop black borders
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = np.column_stack(np.where(gray > 5))
        
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Add small margin
            margin = 10
            y_min = max(0, y_min - margin)
            x_min = max(0, x_min - margin)
            y_max = min(img.shape[0], y_max + margin)
            x_max = min(img.shape[1], x_max + margin)
            
            img = img[y_min:y_max, x_min:x_max]
        
        # Apply subtle enhancement
        img = cv2.bilateralFilter(img, 5, 50, 50)  # Reduced for performance
        
        # Mild color balance
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        return img

# Global stitcher instance
stitcher = ProfessionalImageStitcher()

@app.post("/stitch")
async def stitch_images(files: List[UploadFile] = File(...)):
    """
    Create professional panoramas from multiple images
    """
    try:
        if len(files) < 2:
            raise HTTPException(status_code=400, detail="At least 2 images are required")
        
        if len(files) > 8:  # Reduced limit for better quality
            raise HTTPException(status_code=400, detail="Maximum 8 images allowed for optimal quality")
        
        # Load and validate images
        images = []
        for i, file in enumerate(files):
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
            
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise HTTPException(status_code=400, detail=f"Could not decode {file.filename}")
            
            images.append(img)
            logger.info(f"Loaded image {i+1}: {img.shape[1]}x{img.shape[0]} pixels")
        
        # Perform stitching
        try:
            result = stitcher.stitch_images(images)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        
        if result is None or result.size == 0:
            raise HTTPException(status_code=500, detail="Stitching failed - unable to create panorama")
        
        # Encode result with high quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        success, encoded_img = cv2.imencode('.jpg', result, encode_params)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode result image")
        
        # Return as streaming response
        img_bytes = io.BytesIO(encoded_img.tobytes())
        img_bytes.seek(0)
        
        logger.info(f"Panorama created successfully: {result.shape[1]}x{result.shape[0]} pixels")
        
        return StreamingResponse(
            img_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=panorama.jpg"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stitching failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Professional image stitching service is running"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Professional Image Stitching API",
        "version": "3.1.0",
        "features": [
            "Multi-detector feature detection (SIFT, AKAZE, ORB)",
            "Flexible matching with multiple strategies",
            "Relaxed homography estimation for difficult cases",
            "Feather blending for smooth transitions",
            "Adaptive parameter adjustment",
            "Enhanced error recovery"
        ],
        "endpoints": {
            "/stitch": "POST - Create panorama from multiple images",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)