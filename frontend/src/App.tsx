import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/components/theme-provider";
import { Layout } from "@/components/layout";
import { Toaster } from "sonner";
import Home from "@/pages/Home";
import EvaluationPage from "@/pages/Evaluation";
import ResultsPage from "@/components/results/ResultsPage";
import ProgramSearch from "@/pages/ProgramSearch";
import AboutPage from "@/pages/AboutPage";
import About from "@/pages/About";
import Debug from "@/pages/Debug";

export default function App() {
  return (
    <ThemeProvider defaultTheme="light" storageKey="talent-ai-theme">
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/programs" element={<ProgramSearch />} />
            <Route path="/debug" element={<Debug />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </Layout>
        <Toaster richColors position="top-center" duration={Infinity} closeButton={false} />
      </Router>
    </ThemeProvider>
  );
}
