import { sum, multiply } from "lodash";

/**
 * Class responsible for processing numerical data
 */
export class DataProcessor {
  private cache: Map<string, number[]> = new Map();

  /**
   * Processes an array of numbers by transforming them
   */
  processData(numbers: number[]): number[] {
    const key = numbers.join(",");

    if (this.cache.has(key)) {
      return this.cache.get(key)!;
    }

    const result = this.transform(numbers);
    this.cache.set(key, result);
    return result;
  }

  /**
   * Transforms numbers using various operations
   */
  private transform(numbers: number[]): number[] {
    const doubled = numbers.map((n) => multiply(n, 2));
    const total = sum(doubled);

    // Data structure with anonymous function
    const operations = {
      scale: (arr: number[]) => arr.map((n) => n / total),
    };

    // Call the anonymous function
    return operations.scale(doubled);
  }
}
