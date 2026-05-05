import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# 설정
input_dir = r'C:\ai_image_detection\ai_image_detection\image_data\SDXL'  # 원본 이미지 폴더 경로
output_dir = r'C:\ai_image_detection\ai_image_detection\256_images' # 저장할 폴더 경로
target_size = (256, 256)

# 출력 폴더가 없으면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def resize_image(filename):
    try:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 이미지가 이미 존재하면 건너뛰기 (중단 후 재시작 시 유용)
        if os.path.exists(output_path):
            return
            
        with Image.open(input_path) as img:
            # LANCZOS 필터는 축소 시 화질 저하를 최소화합니다.
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            img_resized.save(output_path, quality=90) # 품질과 용량의 타협점
    except Exception as e:
        print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # 처리할 파일 리스트 추출 (.jpg, .png 등)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"총 {len(files)}장의 이미지 처리를 시작합니다.")
    
    # CPU 코어를 최대한 활용하여 병렬 처리
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(resize_image, files), total=len(files)))

    print("모든 작업이 완료되었습니다!")