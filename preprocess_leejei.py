import os
from PIL import Image

def preprocess_targeted_folders():
    input_root = "C:/ai_image_detection"
    output_root = "C:/processed_256"

   
    target_folders = ["00", "01", "02", "03"]

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    print("🎯 지정된 4개 학회 폴더 전처리를 시작합니다...")

    for folder_name in target_folders:
        # 실제 원본 하위 폴더 경로와 결과물이 들어갈 하위 폴더 경로 설정
        current_input_dir = os.path.join(input_root, folder_name)
        current_output_dir = os.path.join(output_root, folder_name)

        # 혹시 폴더명을 오타 냈거나 없는 경우를 대비한 방어 코드
        if not os.path.exists(current_input_dir):
            print(f"⚠️ 경고: [ {current_input_dir} ] 폴더가 존재하지 않아 건너뜁니다. 이름을 확인하세요!")
            continue

        # 결과물 저장할 하위 폴더 생성
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        # 해당 폴더 안의 이미지 파일만 싹 긁어오기
        file_list = [f for f in os.listdir(current_input_dir) if f.lower().endswith(valid_extensions)]
        print(f"📂 [{folder_name}] 폴더 내 이미지 {len(file_list)}장 작업 시작...")

        processed_count = 0
        for file_name in file_list:
            img_path = os.path.join(current_input_dir, file_name)
            output_path = os.path.join(current_output_dir, file_name)
            
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Center Crop (중앙 정형 자르기)
                    width, height = img.size
                    min_dim = min(width, height)
                    left = (width - min_dim) / 2
                    top = (height - min_dim) / 2
                    right = (width + min_dim) / 2
                    bottom = (height + min_dim) / 2
                    
                    img_cropped = img.crop((left, top, right, bottom))
                    
                    # 256x256 리사이즈
                    img_resized = img_cropped.resize((256, 256), Image.Resampling.LANCZOS)
                    img_resized.save(output_path, 'JPEG', quality=95)
                    
                    processed_count += 1
                    if processed_count % 1000 == 0:
                        print(f" └ ⏳ [{folder_name}] {processed_count}번째 이미지 완료...")

            except Exception as e:
                print(f"❌ {file_name} 처리 실패: {e}")

        print(f"✅ [{folder_name}] 완료! 총 {processed_count}장 전처리됨.\n")

    print("✨ 지정한 모든 학회 데이터셋 전처리가 클린하게 완료되었습니다!")

if __name__ == "__main__":
    preprocess_targeted_folders()