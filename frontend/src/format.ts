/** Display form of an element key: "cu" -> "Cu". Keys stay lowercase everywhere else. */
export const sym = (el: string) => el.slice(0, 1).toUpperCase() + el.slice(1);
