// Converts a lowercase element key into its display symbol.
// For example "cu" becomes "Cu".
// Keys stay lowercase everywhere else.
export const sym = (el: string) => el.slice(0, 1).toUpperCase() + el.slice(1);
