// ===========================
// AgrowAssist — main.js
// ===========================

document.addEventListener("DOMContentLoaded", function () {
  initScrollReveal();
  initNavbarScroll();
});

/* ---- Scroll reveal ---- */
function initScrollReveal() {
  const items = document.querySelectorAll("[data-animate]");
  if (!items.length) return;

  const prefersReduced = window.matchMedia(
    "(prefers-reduced-motion: reduce)",
  ).matches;

  if (prefersReduced) {
    items.forEach((el) => el.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15, rootMargin: "0px 0px -40px 0px" },
  );

  items.forEach((el) => observer.observe(el));
}

/* ---- Navbar scroll state ---- */
function initNavbarScroll() {
  const nav = document.querySelector(".navbar");
  if (!nav) return;

  const toggle = () => {
    nav.classList.toggle("navbar-scrolled", window.scrollY > 20);
  };

  toggle(); // set correct state on page load if not scrolled to top
  window.addEventListener("scroll", toggle, { passive: true });
}
