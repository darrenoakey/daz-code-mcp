import { isEmail } from "validator";

/**
 * Validates if the provided data is a valid user object
 */
export function validateUser(data: any): boolean {
  if (!data || typeof data !== "object") {
    return false;
  }

  const hasRequiredFields =
    typeof data.id === "number" &&
    typeof data.name === "string" &&
    typeof data.email === "string";

  if (!hasRequiredFields) {
    return false;
  }

  return isValidEmail(data.email);
}

/**
 * Checks if an email address is valid
 */
function isValidEmail(email: string): boolean {
  return isEmail(email);
}

/**
 * Formats a user's name to proper case
 */
export function formatUserName(name: string): string {
  return name
    .toLowerCase()
    .split(" ")
    .map((word) => capitalize(word))
    .join(" ");
}

/**
 * Capitalizes the first letter of a word
 */
function capitalize(word: string): string {
  if (!word) return "";
  return word.charAt(0).toUpperCase() + word.slice(1);
}
