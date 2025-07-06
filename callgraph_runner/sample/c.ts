import { validateUser, formatUserName } from "./d";
import axios from "axios";

interface User {
  id: number;
  name: string;
  email: string;
}

/**
 * Service for managing user operations
 */
export class UserService {
  private currentUser: User | null = null;

  /**
   * Gets the current user, fetching from API if needed
   */
  getCurrentUser(): User | null {
    if (!this.currentUser) {
      this.fetchUser();
    }
    return this.currentUser;
  }

  /**
   * Fetches user data from the API
   */
  private async fetchUser(): Promise<void> {
    try {
      const response = await axios.get("/api/user");
      const userData = response.data;

      if (validateUser(userData)) {
        this.currentUser = {
          ...userData,
          name: formatUserName(userData.name),
        };
      }
    } catch (error) {
      console.error("Failed to fetch user:", error);
    }
  }

  /**
   * Updates the current user's name
   */
  updateUserName(newName: string): void {
    if (this.currentUser) {
      this.currentUser.name = formatUserName(newName);
    }
  }
}
