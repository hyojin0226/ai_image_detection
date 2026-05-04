import os
from PIL import Image
from tqdm import tqdm

# 1. 경로 설정
ORIGINAL_FAKE_PATH = r"C:\ai_image_detection\images\fake"
SAVE_PATH = r"C:\ai_image_detection\resized_256\fake"

def run():
    # 저장 폴더 생성
    os.makedirs(SAVE_PATH, exist_ok=True)

    # 2. FLUX 폴더만 타겟팅
    for folder in ['FLUX_DEV', 'FLUX_PRO']:
        in_dir = os.path.join(ORIGINAL_FAKE_PATH, folder)
        out_dir = os.path.join(SAVE_PATH, folder)
        
        if not os.path.exists(in_dir):
            print(f"⚠️ 폴더 없음: {in_dir}")
            continue
            
        os.makedirs(out_dir, exist_ok=True)
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        print(f"🚀 {folder} 처리 시작 ({len(files)}장)")

        # 3. 리사이징 실행
        for f in tqdm(files):
            try:
                with Image.open(os.path.join(in_dir, f)) as img:
                    img.resize((256, 256), Image.Resampling.LANCZOS).convert("RGB").save(os.path.join(out_dir, f), "JPEG")
            except:
                continue

if __name__ == "__main__":
    run()
    print("✅ 전처리 완료!")