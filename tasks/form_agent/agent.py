# from pydantic import BaseModel, ValidationError
# import json
# import os

# DATA_FILE = "data.json"

# # ✅ Updated model with address, no email
# class User(BaseModel):
#     first_name: str
#     last_name: str
#     age: int
#     occupation: str
#     address: str

# # Order of fields in the form
# FIELDS = ["first_name", "last_name", "age", "occupation", "address"]


# def load_data():
#     if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
#         return []  # Start with empty list
#     with open(DATA_FILE, "r") as f:
#         return json.load(f)

# def save_data(data):
#     with open(DATA_FILE, "w") as f:
#         json.dump(data, f, indent=2)

# def main():
#     all_users = load_data()
#     form_data = {}
#     field_index = 0

#     print("Fill in the form. You can overwrite previous fields anytime.")

#     while field_index < len(FIELDS):
#         field = FIELDS[field_index]
#         print(f"\nCurrent form data:")
#         for f in FIELDS:
#             print(f"  {f}: {form_data.get(f, '')}")

#         value = input(f"Enter value for '{field}': ").strip()

#         # If the input looks like "field value", treat it as an update to that field
#         if " " in value and value.split()[0] in FIELDS:
#             update_field = value.split()[0]
#             new_value = " ".join(value.split()[1:])
#             form_data[update_field] = new_value
#             print(f"✅ Updated '{update_field}' to '{new_value}'.")
#             continue

#         # Normal entry
#         form_data[field] = value
#         field_index += 1

#     # ✅ Validation and save
#     try:
#         user = User(**form_data)
#         all_users.append(user.dict())
#         save_data(all_users)
#         print("\n✅ Form submitted successfully!")
#     except ValidationError as e:
#         print("❌ Validation error:", e)

# if __name__ == "__main__":
#     main()

# import requests

# API_BASE = "http://127.0.0.1:8000"

# def create_user():
#     user_data = {}
#     user_data["first_name"] = input("First name: ")
#     user_data["last_name"] = input("Last name: ")
#     user_data["age"] = int(input("Age: "))
#     user_data["occupation"] = input("Occupation: ")
#     user_data["address"] = input("Address: ")

#     res = requests.post(f"{API_BASE}/users", json=user_data)
#     print(res.json())

# def view_users():
#     res = requests.get(f"{API_BASE}/users")
#     users = res.json()
#     for i, user in enumerate(users):
#         print(f"{i}: {user}")
#     return users

# def update_user():
#     users = view_users()
#     if not users:
#         print("No users found.")
#         return

#     idx = int(input("Enter index of user to update: "))
#     if idx < 0 or idx >= len(users):
#         print("Invalid index.")
#         return

#     updated_data = {}
#     for field, value in users[idx].items():
#         new_value = input(f"{field} [{value}]: ") or value
#         if field == "age":
#             new_value = int(new_value)
#         updated_data[field] = new_value

#     res = requests.put(f"{API_BASE}/users/{idx}", json=updated_data)
#     print(res.json())

# def main():
#     while True:
#         print("\n1. Create User")
#         print("2. View Users")
#         print("3. Update User")
#         print("4. Exit")
#         choice = input("Choose: ")
#         if choice == "1":
#             create_user()
#         elif choice == "2":
#             view_users()
#         elif choice == "3":
#             update_user()
#         elif choice == "4":
#             break
#         else:
#             print("Invalid choice.")

# if __name__ == "__main__":
#     main()


# agent.py
import json
import os
from groq import Groq

# ===== CONFIG =====
GROQ_API_KEY = "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"  # Put your API key here
DATA_FILE = "data.json"

# ===== GROQ CLIENT =====
client = Groq(api_key=GROQ_API_KEY)

# ===== JSON HANDLERS =====
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ===== FORM FUNCTIONS =====
def create_entry():
    form_data = {}
    fields = ["name", "age", "email"]

    print("\n--- New Form Entry --- (type 'exit' anytime to quit)")

    for field in fields:
        value = input(f"Enter {field}: ")
        if value.lower() == "exit":
            return
        form_data[field] = value

    print("\nYou entered:")
    for k, v in form_data.items():
        print(f"{k}: {v}")

    confirm = input("Is this correct? (yes/no): ").strip().lower()
    if confirm == "yes":
        data = load_data()
        data.append(form_data)
        save_data(data)
        print("✅ Entry saved successfully!")
    else:
        print("❌ Entry discarded.")

def update_entry():
    data = load_data()
    if not data:
        print("No entries found.")
        return

    for i, entry in enumerate(data, start=1):
        print(f"{i}. {entry}")

    choice = input("Enter entry number to update: ")
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(data):
        print("Invalid choice.")
        return

    idx = int(choice) - 1
    entry = data[idx]

    print("\nCurrent entry:")
    for k, v in entry.items():
        print(f"{k}: {v}")

    field_to_update = input("Enter the field to update (name, age, email): ").strip().lower()
    if field_to_update in entry:
        new_value = input(f"Enter new value for {field_to_update}: ")
        entry[field_to_update] = new_value
        save_data(data)
        print("✅ Entry updated successfully!")
    else:
        print("Invalid field.")

def delete_entry():
    data = load_data()
    if not data:
        print("No entries found.")
        return

    for i, entry in enumerate(data, start=1):
        print(f"{i}. {entry}")

    choice = input("Enter entry number to delete: ")
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(data):
        print("Invalid choice.")
        return

    idx = int(choice) - 1
    deleted = data.pop(idx)
    save_data(data)
    print(f"✅ Deleted entry: {deleted}")

def view_entries():
    data = load_data()
    if not data:
        print("No entries found.")
        return
    print("\n--- All Entries ---")
    for i, entry in enumerate(data, start=1):
        print(f"{i}. {entry}")

# ===== MENU =====
def main():
    while True:
        print("\n=== Form Bot Menu ===")
        print("1. Create new entry")
        print("2. Update entry")
        print("3. Delete entry")
        print("4. View entries")
        print("5. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            create_entry()
        elif choice == "2":
            update_entry()
        elif choice == "3":
            delete_entry()
        elif choice == "4":
            view_entries()
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()

