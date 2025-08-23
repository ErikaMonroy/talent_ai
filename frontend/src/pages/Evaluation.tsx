'use client';

import React from 'react';
import { AssessmentForm } from '@/components/assessment/AssessmentForm';

export default function EvaluationPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        <AssessmentForm />
      </div>
    </div>
  );
}