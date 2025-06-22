import os
import time
from datetime import datetime

while True:
    print(f"\n⏰ Running auto_map_checker.py at {datetime.now().strftime('%H:%M:%S')}")

    # Run the map update script
    os.system("python auto_map_checker.py")

    # Force-add updated files (images + HTML)
    os.system("git add -f annotated_output/*.jpg index.html")

    # Commit even if nothing has changed (empty commit)
    os.system(f'git commit --allow-empty -m "🚀 Auto-update at {datetime.now().isoformat()}"')

    # Push to GitHub
    os.system("git push origin main")

    print("✅ Update pushed. Sleeping for 120 seconds...\n")
    time.sleep(120)
