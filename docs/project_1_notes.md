## Instructions for Collaborators

### A) First Time Use

1. **Get the project**
   Clone the repository:
   ```bash
   git clone https://github.com/sandipan-ds/ai_grievance_system.git
   cd ai_grievance_system
   ```

2. **Create your working branch**
   Always branch off `main`. Name your branch thoughtfully (e.g., `feature/<your-name>`):
   ```bash
   # Ensure you are on main and up to date
   git checkout main
   git pull origin main
   
   # Create and switch to your new branch
   git checkout -b feature/<your-name>
   ```

### B) Regular Use

1. **Make changes locally**
   Edit files naturally.

2. **Stage and commit changes**
   ```bash
   git add .
   git commit -m "Describe your changes"
   ```

3. **Push your branch to GitHub**
   ```bash
   git push -u origin feature/<your-name>
   ```

4. **Create a Pull Request (PR)**
   - Go to the repository on GitHub.
   - Click **Compare & Pull Request**.
   - Base branch: `main`
   - Compare branch: `feature/<your-name>`

5. **Updating your branch (if `main` updates)**
   If another team member merges code into `main` while you are working, pull those updates and merge them into your working branch:
   ```bash
   # Switch to main and get the latest code
   git checkout main
   git pull origin main
   
   # Switch back to your feature branch
   git checkout feature/<your-name>
   
   # Merge the new main into your branch
   git merge main
   ```

### ⚠️ Important Rules
- **Never work or commit directly to `main`.**
- **Always create your own branch for your work.**
- **Always use Pull Requests (PRs) for code review before merging into `main`.**

---

### C) Database Setup (Supabase SQL Dataset)

To securely connect to the project's central Supabase SQL dataset, perform this local setup:

1. Locate the template file at `src/.env.example`.
2. Rename this file to `.env` (simply remove `.example` from the extension).
3. Open the newly renamed `src/.env` file.
4. Look for the `path` variable. Replace `<your_root_foldername>` with the actual path to wherever you cloned this repository on your computer.
