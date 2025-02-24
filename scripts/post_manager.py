import os
import csv
import re
import yaml
from datetime import datetime
import shutil

class PostManager:
    def __init__(self):
        self.posts_dir = '_posts'
        self.images_dir = 'assets/images'
        self.csv_path = 'assets/images/posts.csv'

    def get_next_folder_number(self):
        """다음 폴더 번호를 가져옴 (001, 002 등)"""
        if not os.path.exists(self.csv_path):
            return '001'
        
        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 스킵
            numbers = [row[0] for row in reader]
            if not numbers:
                return '001'
            return f"{int(max(numbers)) + 1:03d}"

    def get_post_info(self, post_file):
        """포스트 파일에서 정보 추출"""
        with open(os.path.join(self.posts_dir, post_file), 'r', encoding='utf-8') as f:
            content = f.read()
            
        # YAML 프론트매터 추출
        if content.startswith('---'):
            _, front_matter, _ = content.split('---', 2)
            metadata = yaml.safe_load(front_matter)
            
        # 이미지 개수 카운트
        image_count = len(re.findall(r'!\[.*?\]\(/assets/images/\d{3}_.*?/\d+\.png\)', content))
            
        return {
            'title': metadata.get('title', '').replace(' ', '_').lower(),
            'date': metadata.get('date', datetime.now()).strftime('%Y-%m-%d'),
            'image_count': image_count
        }

    def create_image_folder(self, folder_number, title):
        """이미지 폴더 생성"""
        folder_path = os.path.join(self.images_dir, f"{folder_number}_{title}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            # 빈 .gitkeep 파일 생성 (빈 폴더도 git에 추가되도록)
            with open(os.path.join(folder_path, '.gitkeep'), 'w') as f:
                pass
        return folder_path

    def update_csv(self, folder_number, title, date, image_count):
        """CSV 파일 업데이트"""
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['folder_number', 'post_title', 'post_date', 'image_count'])

        # 기존 데이터 읽기
        existing_data = {}
        if os.path.exists(self.csv_path):
            with open(self.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_data[row['folder_number']] = row

        # 새 데이터 추가/업데이트
        existing_data[folder_number] = {
            'folder_number': folder_number,
            'post_title': title,
            'post_date': date,
            'image_count': image_count
        }

        # CSV 파일 업데이트
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['folder_number', 'post_title', 'post_date', 'image_count'])
            writer.writeheader()
            writer.writerows(existing_data.values())

    def move_existing_images(self, post_file, folder_number, title):
        """기존 이미지를 새 폴더로 이동하고 포스트 내용 업데이트"""
        post_path = os.path.join(self.posts_dir, post_file)
        new_folder = f"{folder_number}_{title}"
        new_folder_path = os.path.join(self.images_dir, new_folder)
        
        # 포스트 내용 읽기
        with open(post_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 이미지 경로 패턴 매칭
        pattern = r'!\[(.*?)\]\(/assets/images/([^/)]+)\)'
        matches = re.findall(pattern, content)
        
        if matches:
            # 새 폴더 생성
            os.makedirs(new_folder_path, exist_ok=True)
            
            # 각 이미지에 대해
            for idx, (alt_text, img_name) in enumerate(matches, 1):
                old_path = os.path.join(self.images_dir, img_name)
                new_name = f"{idx:02d}{os.path.splitext(img_name)[1]}"
                new_path = os.path.join(new_folder_path, new_name)
                
                # 이미지 파일 이동
                if os.path.exists(old_path):
                    shutil.move(old_path, new_path)
                    
                    # 포스트 내용 업데이트
                    old_img_path = f"/assets/images/{img_name}"
                    new_img_path = f"/assets/images/{new_folder}/{new_name}"
                    content = content.replace(old_img_path, new_img_path)
            
            # 업데이트된 내용 저장
            with open(post_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        return False

    def process_posts(self):
        """모든 포스트 처리"""
        for post_file in os.listdir(self.posts_dir):
            if post_file.endswith('.md'):
                print(f"Processing post: {post_file}")
                post_info = self.get_post_info(post_file)
                folder_number = self.get_next_folder_number()
                
                # 이미지 폴더 생성
                folder_path = self.create_image_folder(folder_number, post_info['title'])
                
                # 기존 이미지 이동 및 포스트 내용 업데이트
                if self.move_existing_images(post_file, folder_number, post_info['title']):
                    print(f"Moved existing images to: {folder_path}")
                
                # CSV 업데이트
                self.update_csv(
                    folder_number,
                    post_info['title'],
                    post_info['date'],
                    post_info['image_count']
                )
                
                print(f"Created image folder: {folder_path}")
                print(f"Updated CSV file: {self.csv_path}")

if __name__ == '__main__':
    manager = PostManager()
    manager.process_posts() 