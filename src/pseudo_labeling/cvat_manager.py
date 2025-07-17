import requests
import json
import os
import glob
import time
from pathlib import Path


class CVATManager:
    """Handles CVAT API operations and annotation management with organization support."""

    def __init__(self, cvat_url="https://cvat2.vbti.nl"):
        self.cvat_url = cvat_url
        self.cvat_api_url = f"{cvat_url}/api"
        self.session = None
        self.current_organization = None

    def authenticate(self, username, password, organization=None):
        """Authenticate with CVAT server and optionally set organization context."""
        self.session = requests.Session()
        resp = self.session.post(f"{self.cvat_api_url}/auth/login",
                                 json={"username": username, "password": password})
        if resp.status_code == 200:
            token = resp.json()["key"]
            self.session.headers.update({"Authorization": f"Token {token}"})

            # Set organization context if provided
            if organization:
                self.set_organization(organization)

            print("✓ CVAT authentication successful.")
            return True
        else:
            raise RuntimeError("CVAT authentication failed. Check credentials.")

    def get_organizations(self):
        """Get list of organizations user has access to."""
        if not self.session:
            raise RuntimeError("Not authenticated. Call authenticate() first.")

        resp = self.session.get(f"{self.cvat_api_url}/organizations")
        if resp.status_code == 200:
            return resp.json()["results"]
        return []

    def set_organization(self, org_name):
        """Set organization context for subsequent API calls."""
        orgs = self.get_organizations()
        org = next((o for o in orgs if o["slug"] == org_name or o["name"] == org_name), None)
        if org:
            self.current_organization = org["id"]
            print(f"Organization set: {org['name']} (ID: {org['id']}, slug: {org['slug']})")

            # Validate access
            if self.validate_organization_access():
                print("Organization access confirmed")
            else:
                print("Warning: Organization access validation failed")

            return True
        else:
            available = [f"{o['name']} (ID: {o['id']}, slug: {o['slug']})" for o in orgs]
            print(f"✗ Organization '{org_name}' not found")
            print(f"Available organizations: {available}")
            return False

    def validate_organization_access(self):
        """Validate organization access using only working methods and cache the best one."""
        if not self.current_organization:
            print("✓ Using personal workspace (no organization)")
            return True

        # Check if organization exists and is accessible
        resp = self.session.get(f"{self.cvat_api_url}/organizations/{self.current_organization}")
        if resp.status_code != 200:
            print(f"✗ Organization ID {self.current_organization} not accessible: {resp.status_code}")
            return False

        org_data = resp.json()
        org_slug = org_data.get('slug')
        print(f"✓ Organization validated: {org_data['name']} (ID: {self.current_organization})")

        # Test only working methods in priority order
        working_methods = [
            {"params": {"org": org_slug}, "name": "org_slug", "type": "param"},
            {"params": {"org_id": self.current_organization}, "name": "org_id", "type": "param"},
            {"headers": {"X-Organization": org_slug}, "name": "x_org_slug", "type": "header"},
        ]

        for method in working_methods:
            if not org_slug and "org_slug" in method["name"]:
                continue  # Skip slug methods if no slug available

            test_resp = self.session.get(
                f"{self.cvat_api_url}/projects",
                params=method.get("params", {}),
                headers=method.get("headers", {}),
                timeout=10
            )

            if test_resp.status_code == 200:
                # Cache the working method for later use
                self.preferred_method = method
                print("✓ Organization access confirmed")
                return True

        print("✗ No working organization access method found")
        return False

    def get_or_create_project(self, project_name, dataset_labels, project_id=None):
        """Optimized project handling using cached working methods."""
        # Option 1: Use specific project ID if provided
        if project_id:
            resp = self.session.get(f"{self.cvat_api_url}/projects/{project_id}")
            if resp.status_code == 200:
                project_data = resp.json()
                print(f"Using existing CVAT project: {project_data['name']} (ID: {project_id})")
                return project_id
            else:
                print(f"Warning: Project ID {project_id} not found or not accessible")

        # Option 2: Search by name using preferred method if available
        if hasattr(self, 'preferred_method') and self.current_organization:
            method = self.preferred_method
            print(f"Searching for project using cached method: {method['name']}")

            resp = self.session.get(
                f"{self.cvat_api_url}/projects",
                params=method.get("params", {}),
                headers=method.get("headers", {}),
                timeout=10
            )

            if resp.status_code == 200:
                projects = resp.json()["results"]
                for project in projects:
                    if project["name"] == project_name:
                        print(f"✓ Found existing project: {project_name} (ID: {project['id']})")
                        return project["id"]
                print(f"Project '{project_name}' not found, will create new one")
            else:
                print(f"Project search failed: {resp.status_code}")

        # Fallback: search without organization context
        if not hasattr(self, 'preferred_method') or not self.current_organization:
            resp = self.session.get(f"{self.cvat_api_url}/projects")
            if resp.status_code == 200:
                projects = resp.json()["results"]
                for project in projects:
                    if project["name"] == project_name:
                        print(f"Found existing project in personal space: {project_name} (ID: {project['id']})")
                        return project["id"]

        # Option 3: Create new project using preferred method
        labels = [{"name": label, "attributes": []} for label in dataset_labels]
        payload = {"name": project_name, "labels": labels}

        if hasattr(self, 'preferred_method') and self.current_organization:
            method = self.preferred_method
            print(f"Creating project using cached method: {method['name']}")

            resp = self.session.post(
                f"{self.cvat_api_url}/projects",
                json=payload,
                params=method.get("params", {}),
                headers=method.get("headers", {}),
                timeout=30
            )

            if resp.status_code == 201:
                pid = resp.json()["id"]
                print(f"✓ Created project in organization: {project_name} (ID: {pid})")
                return pid
            else:
                print(f"Organization project creation failed: {resp.status_code}")

        # Fallback: create in personal space
        print("Creating project in personal workspace...")
        resp = self.session.post(f"{self.cvat_api_url}/projects", json=payload)

        if resp.status_code == 201:
            pid = resp.json()["id"]
            print(f"Created project in personal workspace: {project_name} (ID: {pid})")
            return pid
        else:
            raise RuntimeError(f"CVAT project creation failed: {resp.status_code} - {resp.text}")

    def create_task(self, project_id, task_name):
        """Create a new CVAT task using cached organization method."""
        payload = {
            "name": task_name,
            "project_id": project_id,
            "mode": "annotation",
            "overlap": 0,
            "segment_size": 0,
            "dimension": "2d"
        }

        # Use cached method if available (same as project creation)
        if hasattr(self, 'preferred_method') and self.current_organization:
            method = self.preferred_method
            resp = self.session.post(
                f"{self.cvat_api_url}/tasks",
                json=payload,
                params=method.get("params", {}),
                headers=method.get("headers", {}),
                timeout=30
            )
        else:
            # Fallback for personal workspace
            resp = self.session.post(f"{self.cvat_api_url}/tasks", json=payload)

        if resp.status_code == 201:
            tid = resp.json()["id"]
            print(f"✓ Task created: {task_name} (ID: {tid})")
            return tid
        else:
            raise RuntimeError(f"CVAT task creation failed: {resp.text}")

    def upload_images(self, task_id, image_dir):
        """Upload images to CVAT task and wait for processing to complete."""
        url = f"{self.cvat_api_url}/tasks/{task_id}/data"
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.[jp][pn]g")) +
                             glob.glob(os.path.join(image_dir, "*.jpeg")))
        if not image_files:
            raise FileNotFoundError(f"No images found in folder: {image_dir}")

        print(f"Uploading {len(image_files)} images to CVAT...")
        files = {
            f"client_files[{i}]": (os.path.basename(path), open(path, "rb"), "image/jpeg")
            for i, path in enumerate(image_files)
        }

        payload = {"image_quality": 70, "sorting_method": "lexicographical"}

        # Use cached method if available (same as task creation)
        if hasattr(self, 'preferred_method') and self.current_organization:
            method = self.preferred_method
            resp = self.session.post(
                url,
                files=files,
                data=payload,
                params=method.get("params", {}),
                headers=method.get("headers", {}),
                timeout=120
            )
        else:
            # Fallback for personal workspace
            resp = self.session.post(url, files=files, data=payload)

        # Close file handles
        for f in files.values():
            f[1].close()

        if resp.status_code != 202:
            raise RuntimeError(f"CVAT image upload failed: {resp.text}")

        rq_id = resp.json().get("rq_id")
        print(f"Images upload started. Request ID: {rq_id}")

        # Wait for image upload to complete
        self._wait_for_data_upload(task_id, rq_id)
        print("Images uploaded and processed successfully.")

    def _wait_for_data_upload(self, task_id, rq_id):
        """Wait for data upload to complete by checking task status."""
        print("Waiting for image processing to complete...")
        max_attempts = 60  # 5 minutes max wait (5 seconds * 60)
        attempt = 0

        while attempt < max_attempts:
            # Check the task status instead of request status
            resp = self.session.get(f"{self.cvat_api_url}/tasks/{task_id}")
            if resp.status_code != 200:
                print(f"Failed to check task status: {resp.text}")
                break

            task_data = resp.json()
            # Check if the task has data (images have been processed)
            if task_data.get("size", 0) > 0:
                print(f"Image processing complete. Task contains {task_data['size']} images.")
                return

            # Also check the original request status
            req_resp = self.session.get(f"{self.cvat_api_url}/requests/{rq_id}")
            if req_resp.status_code == 200:
                req_state = req_resp.json().get("state")
                if req_state == "failed":
                    raise RuntimeError("Image upload failed.")
                elif req_state == "finished":
                    # Double-check that images are actually loaded
                    time.sleep(2)  # Give CVAT a moment to update
                    continue

            print(f"Still processing... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(5)
            attempt += 1

        if attempt >= max_attempts:
            raise RuntimeError("Timeout waiting for image upload. Check CVAT manually.")

    def upload_annotations(self, task_id, annotation_file):
        """Upload COCO annotations to CVAT task using proper file upload format."""
        format_name = "COCO 1.0"

        print(f"Uploading annotations to CVAT...")

        try:
            with open(annotation_file, "rb") as f:
                # Prepare the multipart form data
                files = {"annotation_file": ("instances_default.json", f, "application/json")}
                data = {
                    "format": format_name,
                    "location": "local"
                }

                # Use cached method if available (same as other operations)
                if hasattr(self, 'preferred_method') and self.current_organization:
                    method = self.preferred_method
                    # Add organization params to data instead of separate params
                    if method.get("params"):
                        data.update(method["params"])

                    headers = method.get("headers", {})
                    resp = self.session.post(
                        f"{self.cvat_api_url}/tasks/{task_id}/annotations",
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=60
                    )
                else:
                    # Fallback for personal workspace
                    resp = self.session.post(
                        f"{self.cvat_api_url}/tasks/{task_id}/annotations",
                        files=files,
                        data=data,
                        timeout=60
                    )

                if resp.status_code == 202:
                    rq_id = resp.json().get("rq_id")
                    print(f"✓ Annotation upload started (Request ID: {rq_id})")
                    self._check_annotation_status(rq_id)
                else:
                    print(f"Annotation upload failed: {resp.status_code} - {resp.text}")

        except Exception as e:
            print(f"Exception during annotation upload: {e}")
            print("Please manually upload annotations in CVAT.")

    def _check_annotation_status(self, rq_id):
        """Check CVAT annotation upload status with cleaner output."""
        url = f"{self.cvat_api_url}/requests/{rq_id}"
        max_attempts = 24  # 2 minutes max wait
        attempt = 0

        while attempt < max_attempts:
            resp = self.session.get(url)
            if resp.status_code != 200:
                print(f"✗ Failed to check status: {resp.status_code}")
                break

            status_data = resp.json()
            state = status_data.get("state")

            if state == "finished":
                print("✓ Annotation upload completed successfully")
                break
            elif state == "failed":
                error_msg = status_data.get("message", "Unknown error")
                print(f"✗ Annotation upload failed: {error_msg}")
                break
            else:
                if attempt % 4 == 0:  # Only print every 4th attempt to reduce spam
                    print(f"Processing annotations... ({attempt + 1}/{max_attempts})")
                time.sleep(5)
                attempt += 1

        if attempt >= max_attempts:
            print("⚠ Timeout waiting for annotation upload. Check CVAT manually.")